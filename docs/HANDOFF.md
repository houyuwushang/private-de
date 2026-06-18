# Handoff

## Repository Sync - 2026-06-18

Task:

- Sync the current QDTE research/code state to the GitHub repository so the next
  work can continue from a clean remote checkpoint.

Changed Files:

- Documentation and experiment notes:
  - `README.md`
  - `architecture.md`
  - `architecture_zh.md`
  - `docs/HANDOFF.md`
  - `docs/QDTE_ABLATION_SUMMARY.md`
  - `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- Configs:
  - `configs/acs_qdte.yaml`
  - `configs/adult_qdte.yaml`
  - `configs/adult_qdte_gpu_highpower.yaml`
  - `configs/adult_qdte_gpu_highpower_directed_group.yaml`
  - `configs/smoke.yaml`
- Core implementation:
  - `qdte/config_validation.py`
  - `qdte/eval/metrics.py`
  - `qdte/evolution/engine.py`
  - `qdte/evolution/gpu_candidates.py`
  - `qdte/evolution/transport.py`
  - `qdte/measurement/measure.py`
  - `qdte/queries/workload.py`
- Population driver scripts:
  - `scripts/population_worker.py`
  - `scripts/run_population.py`
- Tests:
  - `tests/test_config_validation.py`
  - `tests/test_engine_smoke.py`
  - `tests/test_gpu_candidates.py`
  - `tests/test_measurement.py`
  - `tests/test_metrics.py`
  - `tests/test_transport.py`
  - `tests/test_workload.py`

Tests Run:

- `git diff --check`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `183 passed in 22.83s`.

Current Status:

- Single-layer QDTE mainline is currently:
  orthogonal measurement/workload, feasible consistency projection,
  unweighted objective, and constructive A-style directed edits.
- Group transport is implemented and useful for measured-loss reduction, but
  the latest orthogonal smoke comparison keeps standalone A as the best
  offline true-RMSE result.
- Adaptive select-measure-generate is intentionally paused. Do not implement a
  generic exponential-mechanism selection rule until the paper/documented
  quality function and budget split are added.

Next Recommended Task:

- After the adaptive query-selection definition is written, implement
  `privacy.measurement_mode=adaptive_select_measure` exactly from that
  definition, preserving the DP boundary and adding a matching ablation.

## Loss-Scale Comparison Clarification - 2026-06-18

Task:

- Clarify why recent current-measurement runs have unweighted loss around
  `100+` while earlier notes reported unweighted loss around `21`.
- Correct the comparison after noticing that the latest A+group run reused an
  original `mixed` measurement artifact instead of the orthogonal mainline.

Clarification:

- The `21`-level result was not the same experimental condition as the latest
  A+group comparison.
- Main `21`-level DP runs:
  - `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair`:
    orthogonal mixed workload, `242` queries, unweighted objective,
    final unweighted loss `21.849774742`, true RMSE `0.002279082`;
  - `outputs/exp_A_orthogonal_mixed_projaware_varonly16_consistency2000_constructive_pair`:
    orthogonal mixed workload, `242` queries, projection-aware variance,
    final unweighted loss `21.917599106`, true RMSE `0.002305222`.
- Latest current-measurement A+group run:
  - `outputs/exp_A_original_mixed_plus_group_on_current_measurement_2000`;
  - original/current mixed workload, `243` queries;
  - reused `outputs/exp_jaxactive2000_core_single_query` measurement;
  - variance-weighted objective;
  - final unweighted loss `111.946766719`, true RMSE `0.003331481`.
- Therefore the `21` vs `100+` gap is mostly a comparison-context issue:
  orthogonal grouped measurement/workload plus different objective/variance
  setting vs the older current mixed measurement artifact.
- There was also an oracle/exact non-DP probe with losses `21 -> 12`, but that
  is a separate non-DP sanity check and should not be compared to DP measured
  runs as a privacy-preserving result.

Corrected Orthogonal A+Group Rerun:

- Output:
  `outputs/exp_unweightedobj_orthogonal_plus_group_consistency2000_constructive_pair`.
- Config source:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`.
- Reused measurement:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/measurements.json`.
- Workload: orthogonal mixed, `242` queries.
- Objective: `qdte.objective_weighting=unweighted`.
- Final unweighted loss `20.178445721`.
- Final true RMSE `0.002313275`.
- Accepted edits `1041`.

Corrected Orthogonal Pure/Adaptive Group Reruns:

- Pure group output:
  `outputs/exp_unweightedobj_orthogonal_pure_group_consistency2000`.
- Edit + adaptive group output:
  `outputs/exp_unweightedobj_orthogonal_edit_plus_adaptive_group_consistency2000`.
- Both used:
  - config source:
    `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`;
  - reused measurement:
    `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/measurements.json`;
  - workload: orthogonal mixed, `242` queries;
  - objective: `qdte.objective_weighting=unweighted`;
  - `max_iters=2000`, `total_candidates_per_iter=256`,
    `accepted_per_iter=8`, `num_candidates_scored=512000`.

Orthogonal Same-Measurement Comparison:

| Method | Transport | Final unweighted loss | Offline true RMSE | Accepted edits |
|---|---|---:|---:|---:|
| A / `constructive_pair` | `constructive_pair` | `21.849774742` | `0.002279082` | `955` |
| A + group augment | `constructive_pair` + group augment | `20.178445721` | `0.002313275` | `1041` |
| Pure group | `directed_group`, no positive fill | `18.563489342` | `0.002327521` | `1027` |
| Edit + adaptive group | positive fill + adaptive group augment | `17.938319349` | `0.002313275` | `1036` |

Interpretation:

- The `100+` loss should not be used as evidence about the orthogonal mainline.
- Under the intended orthogonal measurement setting, A+group remains in the
  `20`-level unweighted-loss regime and slightly improves measured loss over A.
- Pure group and edit+adaptive group reduce measured loss further in the same
  setting; edit+adaptive group is the best by measured loss in this smoke run.
- Offline true RMSE does not follow measured loss monotonically:
  standalone A remains best by true RMSE in this single run, while A+group and
  edit+adaptive group tie at `0.002313275`.
- Current read: group transport is real and useful for matching the
  noisy/projected orthogonal target, but the constructive A story remains
  cleaner for true utility and for the individual-level directed-edit
  narrative. Multi-seed results should decide whether group becomes a main
  method or stays as an ablation/second-paper direction.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Read existing `metrics_final.json`, `workload_summary.json`, and
  `config_resolved.yaml` files.
- Ran the corrected orthogonal A+group experiment above.
- Ran corrected orthogonal pure group and edit+adaptive group experiments.
- Run `git diff --check` before closing the task.

Current Status:

- Low `21` unweighted loss remains a real result, but it belongs to the
  orthogonal mixed measurement setting, not the latest current mixed A+group
  comparison.
- Corrected orthogonal A+group run confirms the mainline is still around
  `20` unweighted loss.
- Corrected orthogonal group runs are complete and show lower measured loss
  than A/A+group, with slightly worse offline true RMSE than standalone A.

Next Recommended Task:

- Run multi-seed orthogonal experiments for A, A+group, pure group, and
  edit+adaptive group. Decide the paper narrative from mean/std true utility
  and measured loss, not a single smoke run.

## Constructive Pair Plus Group Augment - 2026-06-18

Task:

- Answer whether `constructive_pair` and group advantage are mutually exclusive.
- Implement a fair `constructive_pair + group` combination and compare it under
  the same current DP measurement artifact.

What Changed:

- Added optional `qdte.constructive_pair_group_augment`.
- The `constructive_pair` transport now runs the constructive A-style selection
  first, then can run one directed-group augment on the remaining edit budget.
- The augment blocks rows already accepted by constructive pair, scores against
  the virtual residual after the constructive batch, combines the selected
  deltas, and applies one final QDTE state update.
- Added adaptive/loss/noise-floor/plateau controls mirroring the directed-group
  positive-fill augment controls.
- Added timeseries/runtime/final-metric diagnostics for the constructive-pair
  group augment path.
- Added config validation coverage for the new controls.

Changed Files:

- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_config_validation.py tests/test_engine_smoke.py tests/test_transport.py -q`
  - Result: `88 passed in 4.18s`.

Experiment Run:

- Output:
  `outputs/exp_A_original_mixed_plus_group_on_current_measurement_2000`.
- Config source:
  `outputs/exp_A_original_mixed_consistency2000_constructive_pair/config_resolved.yaml`.
- Reused current measurement:
  `outputs/exp_jaxactive2000_core_single_query/measurements.json`.
- Common budget: `max_iters=2000`, `total_candidates_per_iter=256`,
  `accepted_per_iter=8`, `num_candidates_scored=512000`.

Fair Same-Measurement Comparison:

| Method | Weighted measured loss | Unweighted measured loss | Offline true RMSE | Accepted edits |
|---|---:|---:|---:|---:|
| A / `constructive_pair` | `2.254667713750168` | `146.2320195284455` | `0.003346271188083466` | not re-read |
| A + group augment | `1.9123104370472743` | `111.94676671892095` | `0.003331480966792211` | `1639` |
| Pure group advantage | `1.880857559037812` | `104.86654506593634` | `0.0032729091274245` | not re-read |
| Edit + adaptive group | `1.8368689239998532` | `102.95992265611578` | `0.0032482980665601035` | not re-read |

Interpretation:

- `constructive_pair` and group advantage are not algorithmically mutually
  exclusive.
- They were previously mutually exclusive only in the engineering shape of the
  `transport_mode` branches.
- The combined A+group run improves over A alone on the same measurement:
  weighted loss `2.2547 -> 1.9123`, unweighted loss `146.23 -> 111.95`, and
  offline true RMSE `0.003346 -> 0.003331`.
- The current A+group implementation still does not beat pure group or
  edit+adaptive group in this smoke setting. This suggests the combination is
  useful but the group stage is not yet adding more than the standalone group
  search can find.

Current Status:

- Implementation complete.
- Smoke comparison complete.
- DP boundary preserved: the run reused noisy/projected measurements for
  optimization; exact true answers were computed only as offline evaluation
  metrics.

Next Recommended Task:

- Compare A, pure group, edit+adaptive group, and A+group over multiple seeds
  with the same measurement construction and report mean/std. If A+group is not
  consistently better, keep it as an ablation rather than the main method.

## Original-Candidate Group Advantage Comparison - 2026-06-18

Task:

- Re-check group advantage under the original small candidate budget instead of
  the high-power GPU budget.
- Address the concern that million-candidate experiments mainly measure search
  coverage/GPU throughput rather than algorithmic value.

Setup:

- Base config:
  `outputs/exp_jaxactive2000_core_single_query/config_resolved.yaml`.
- Reused the same DP measurement artifact for all runs:
  `outputs/exp_jaxactive2000_core_single_query/measurements.json`.
- Common budget:
  - `max_iters=2000`;
  - `total_candidates_per_iter=256`;
  - `accepted_per_iter=8`;
  - `num_candidates_scored=512000`;
  - true-query evaluation disabled for speed; measured-loss comparison only.

Results:

- Loss/RMSE metric definitions for this section:
  - `weighted loss` is `final_measured_loss`, i.e.
    `0.5 * sum_q residual[q]^2 * inv_variance[q]`;
  - `std RMSE` is `sqrt(2 * weighted_loss / num_queries)`;
  - `unweighted loss` is `0.5 * sum_q residual[q]^2`;
  - `unweighted RMSE` is `sqrt(2 * unweighted_loss / num_queries)`;
  - `true RMSE` is offline exact-query error against the real data and was
    computed only after optimization, never used by QDTE selection/scoring.
- Edit advantage only:
  - output: `outputs/exp_origcand2000_edit_advantage_current`;
  - offline true-eval output:
    `outputs/exp_origcand2000_edit_advantage_current_trueeval`;
  - transport: `microbatch_greedy`;
  - final measured loss `7.050950126781752`;
  - final unweighted measured loss `680.3885696974367`;
  - final standardized measured RMSE `0.24089938009786793`;
  - final unweighted residual RMSE `2.3664120358542156`;
  - offline true-query RMSE `0.004102593779569775`;
  - accepted edits `1500`.
- Pure group advantage:
  - output: `outputs/exp_origcand2000_pure_group_advantage_current`;
  - offline true-eval output:
    `outputs/exp_origcand2000_pure_group_advantage_current_trueeval`;
  - transport: `directed_group`;
  - `directed_group_positive_fill=false`;
  - final measured loss `1.880857559037812`;
  - final unweighted measured loss `104.86654506593634`;
  - final standardized measured RMSE `0.12441989040295456`;
  - final unweighted residual RMSE `0.9290312892175134`;
  - offline true-query RMSE `0.0032729091274245`;
  - accepted edits `1556`.
- Edit advantage plus adaptive group repair:
  - output: `outputs/exp_origcand2000_edit_plus_adaptive_group_current`;
  - offline true-eval output:
    `outputs/exp_origcand2000_edit_plus_adaptive_group_current_trueeval`;
  - transport: `directed_group`;
  - `directed_group_positive_fill=true`;
  - `directed_group_positive_fill_augment=true`;
  - `directed_group_positive_fill_augment_trigger=adaptive`;
  - final measured loss `1.8368689239998532`;
  - final unweighted measured loss `102.95992265611578`;
  - final standardized measured RMSE `0.1229563448910628`;
  - final unweighted residual RMSE `0.9205469954501121`;
  - offline true-query RMSE `0.0032482980665601035`;
  - accepted edits `1741`.

Comparison To Previous Strong 243-Query Smoke Results:

| Method | Weighted loss | Std RMSE | Unweighted loss | Unweighted RMSE | Offline true RMSE | True MAE | Candidates | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| New edit advantage | `7.05095` | `0.240899` | `680.389` | `2.36641` | `0.00410259` | `0.00254321` | `512000` | `1500` |
| New pure group advantage | `1.88086` | `0.124420` | `104.867` | `0.929031` | `0.00327291` | `0.00226749` | `512000` | `1556` |
| New edit plus adaptive group | `1.83687` | `0.122956` | `102.960` | `0.920547` | `0.00324830` | `0.00218930` | `512000` | `1741` |
| Old single query | `5.82225` | `0.218906` | `541.649` | `2.11140` | `0.00387935` | `0.00248148` | `512000` | `1461` |
| Old constructive anneal32 measured-loss best | `1.93274` | `0.126124` | `109.588` | `0.949716` | `0.00329546` | `0.00224280` | `512000` | `1750` |
| Old bounded best partner | `1.99748` | `0.128219` | `116.462` | `0.979050` | `0.00323497` | `0.00221811` | `520994` | `1169` |
| Old A original mixed true-RMSE strong | `2.27805` | `0.136929` | `77.1327` | `0.796767` | `0.00272543` | `0.00202058` | `512000` | `1456` |

Interpretation:

- Under the original small candidate budget, group advantage is clearly useful:
  `1.88` vs `7.05` measured loss against edit advantage only.
- The combined edit-plus-adaptive-group version is slightly better than pure
  group advantage on this smoke workload: `1.837` vs `1.881`.
- By weighted measured loss and standardized measured RMSE, the new adaptive
  group result is the best among the same 243-query smoke comparisons above.
- By offline true-query RMSE, some previous constructive variants remain
  stronger; for example `A_original_mixed` has true RMSE `0.00272543` while the
  new adaptive group result has true RMSE `0.00324830`. This is expected because
  QDTE optimizes noisy/projected measurements, not true answers.
- This changes the prior read from the high-power experiments. The
  million-candidate setting is useful for throughput and late-stage adult runs,
  but it can hide the algorithmic value of group advantage because huge
  candidate coverage already makes many single edits available.
- For paper ablations, report both:
  - small-candidate regime: demonstrates group advantage as an algorithmic
    search improvement;
  - high-candidate regime: demonstrates scalability and GPU throughput, but is
    less clean as evidence for group advantage itself.

High-Candidate Context:

- Adult high-power setting uses `1,572,864` candidates per iteration.
- 500-step positive-fill baseline:
  - output: `outputs/adult_qdte_gpu_highpower_directed_group_500iter`;
  - loss `5.436e8 -> 3.765e6`;
  - final RMS unweighted residual `29.51` counts/query;
  - rough unweighted raw-noise floor is about `1.922e6`, so final loss is about
    `1.96x` that scale.
- 100-step continuation from that checkpoint with adaptive group repair:
  - output: `outputs/adult_qdte_gpu_highpower_directed_group_continue100_lossgate`;
  - loss `3.765e6 -> 2.615e6`;
  - final loss is about `1.36x` the same rough noise-floor scale.

Current Judgment:

- The original-candidate result is the stronger evidence for group advantage.
- The high-power result is evidence that the implementation can spend large
  candidate budgets effectively and approach the measurement noise scale, but
  it should not be used alone to argue the group advantage innovation.

Additional Probes Run:

- Offline true-query evaluation with `max_iters=0` for:
  - `outputs/exp_origcand2000_edit_advantage_current_trueeval`;
  - `outputs/exp_origcand2000_pure_group_advantage_current_trueeval`;
  - `outputs/exp_origcand2000_edit_plus_adaptive_group_current_trueeval`;
  - `outputs/exp_jaxactive2000_core_single_query_trueeval_current_metrics`;
  - `outputs/exp_constructive_pair_accept_anneal32_2000_trueeval_current_metrics`;
  - `outputs/exp_A_original_mixed_consistency2000_trueeval_current_metrics`;
  - `outputs/exp_cd_consistency_2000_bounded_best_partner_trueeval_current_metrics`.
- `git diff --check`
  - Result: passed.

Next Recommended Task:

- Run the original-candidate comparison on multiple seeds and then repeat on
  adult with `total_candidates_per_iter=4096` to bridge the small smoke setting
  and the high-power adult setting.

## Directed-Group Search Efficiency Gate - 2026-06-18

Task:

- Optimize directed-group usage for search/evolution efficiency rather than
  treating GPU power draw as the main success metric.
- Keep group search in the algorithm; do not remove it just because a dense
  group augment variant was inefficient.

What Changed:

- Added `qdte.directed_group_positive_fill_augment_max_loss`.
  - `0` disables the loss gate.
  - Positive values allow group augment only when current measured loss is at or
    below that threshold.
- Updated the high-power directed-group config to use group as late-stage
  plateau repair:
  - `directed_group_positive_fill=true`;
  - `directed_group_positive_fill_augment=true`;
  - `directed_group_positive_fill_augment_threshold=64`;
  - `directed_group_positive_fill_augment_max_loss=4000000.0`;
  - lightweight augment pool remains `seed_count=16`, `max_pool=512`,
    `max_size=8`.
- Added efficiency metrics to `metrics_final.json` and `runtime.json`:
  - `loss_reduction_per_million_candidates`;
  - `unweighted_loss_reduction_per_second`;
  - `unweighted_loss_reduction_per_million_candidates`;
  - `scoring_time_fraction`;
  - `transport_time_fraction`.
- Added timeseries diagnostics for the new group loss gate:
  - `directed_group_positive_fill_augment_loss_gate_passed`.

Changed Files:

- `configs/adult_qdte_gpu_highpower_directed_group.yaml`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_transport.py tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - Result: `87 passed in 4.25s`.
- `git diff --check`
  - Result: passed.

Efficiency Experiments:

- 500-step positive-fill baseline:
  - output: `outputs/adult_qdte_gpu_highpower_directed_group_500iter`;
  - final measured loss `3.76538e6`;
  - accepted edits `73371`;
  - generation time `180.017s`;
  - transport fraction `0.0499`;
  - loss reduction per million candidates `686465.24`.
- 500-step dense-ish early augment, threshold `64`, no loss gate:
  - output:
    `outputs/adult_qdte_gpu_highpower_directed_group_lightaugment_500iter`;
  - final measured loss `4.06679e6`;
  - interpretation: group fires too early and hurts full-run convergence.
- 500-step stricter augment, threshold `16`, no loss gate:
  - output:
    `outputs/adult_qdte_gpu_highpower_directed_group_lightaugment16_500iter`;
  - final measured loss `3.93377e6`;
  - interpretation: too conservative to recover the useful late-stage group
    effect.
- 500-step loss-gated augment, threshold `64`, max loss `4.0e6`:
  - output:
    `outputs/adult_qdte_gpu_highpower_directed_group_lossgate_500iter`;
  - final measured loss `3.80631e6`;
  - accepted edits `73214`;
  - generation time `180.819s`;
  - transport fraction `0.0543`;
  - loss reduction per million candidates `686413.21`;
  - group gate was closed at logged iterations `1/100/200/300/400` and open
    by iteration `500`, where it accepted `8` group edits.
- 100-step continuation from the same 500-step baseline checkpoint, no augment:
  - output:
    `outputs/adult_qdte_gpu_highpower_directed_group_continue100_noaugment`;
  - initial loss `3.76538e6`;
  - final loss `2.85356e6`;
  - loss reduction per second `22530.21`;
  - loss reduction per million candidates `5797.22`;
  - accepted edits `3001`.
- 100-step continuation from the same checkpoint, loss-gated group augment:
  - output:
    `outputs/adult_qdte_gpu_highpower_directed_group_continue100_lossgate`;
  - initial loss `3.76538e6`;
  - final loss `2.61517e6`;
  - loss reduction per second `27540.87`;
  - loss reduction per million candidates `7312.86`;
  - accepted edits `3904`;
  - transport fraction increased from `0.0431` to `0.0653`, but late-stage
    search efficiency improved materially.

Current Status:

- GPU power draw is only a hardware health signal. It is not the right metric
  for algorithm quality.
- Search/evolution efficiency should be judged by loss reduction per wall time,
  per candidate, and per transport overhead.
- Current evidence:
  - directed group is useful as late-stage plateau repair;
  - letting group augment fire throughout the run is harmful;
  - a loss gate removes most of the mid-stage harm and preserves the useful
    continuation behavior;
  - on a strict 500-step budget, positive-fill alone is still slightly better
    than the current gated group (`3.765e6` vs `3.806e6`);
  - after the 500-step checkpoint, gated group is clearly better over the next
    100 steps (`2.615e6` vs `2.854e6`).

Next Recommended Task:

- Replace the hand-tuned `4.0e6` loss gate with an adaptive plateau trigger:
  use recent loss-reduction rate, prefix accepted count, and/or noise-floor
  ratio so group repair starts when single-edit positive-fill has actually
  stalled.
- Improve group construction quality next, not raw GPU utilization. The next
  useful target is higher late-stage loss reduction per transport second while
  keeping the group pool small.

## Directed-Group GPU Backend And Positive-Fill Throughput Fix - 2026-06-18

Task:

- Connect `directed_group` to the JAX/GPU transport path.
- Diagnose why a high-throughput run scored tens of millions of candidates
  but accepted only `8` edits per iteration.
- Preserve the QDTE objective invariants and DP boundary.

What Changed:

- Added a JAX-backed directed-group transport path:
  - `qdte.directed_group_backend: cpu|jax|gpu`;
  - `gpu` is currently an alias for the JAX implementation.
- Added `directed_group_positive_fill` in the engine:
  - when enabled, `directed_group` first accepts a normal positive-advantage
    nonconflicting batch using the existing prefix objective;
  - if that positive batch accepts nothing, it falls back to directed-group
    compensating search;
  - this fixes the early-stage throughput bug where `accepted_per_iter=1024`
    was effectively capped by `directed_group_max_size=8`.
- Added optional `directed_group_positive_fill_augment`:
  - default is `false`;
  - when enabled, QDTE accepts the positive prefix, computes a virtual residual,
    blocks already accepted rows, and then tries one directed-group augment on
    the remaining candidate pool;
  - the real QDTE state is still updated once, after the combined transport is
    selected.
- Added diagnostics to `metrics_final.json`, `runtime.json`, and
  `metrics_timeseries.csv` for:
  - `directed_group_backend`;
  - `directed_group_positive_fill`;
  - `directed_group_positive_fill_selected`;
  - `directed_group_positive_fill_accepted`;
  - `directed_group_positive_fill_augment_attempted`;
  - `directed_group_positive_fill_augment_accepted`;
  - `directed_group_positive_fill_total_accepted`.

Changed Files:

- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `configs/adult_qdte_gpu_highpower_directed_group.yaml`
- `tests/test_transport.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_transport.py tests/test_config_validation.py -q`
  - Result: `80 passed in 1.25s`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_transport.py tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - Result: `87 passed in 4.21s`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -c "from qdte.config import load_yaml; from qdte.config_validation import validate_config; cfg=load_yaml('configs/adult_qdte_gpu_highpower_directed_group.yaml'); validate_config(cfg); print(cfg['qdte']['transport_mode'], cfg['qdte']['directed_group_backend'], cfg['qdte']['directed_group_positive_fill'])"`
  - Result: `directed_group jax True`.

Key Experiments:

- Exact-target smoke JAX directed-group probe:
  - output: `outputs/exp_oracle_exact_smoke_directed_group_jax_probe`;
  - started from A-wide loss `21`;
  - final measured/unweighted loss `12`;
  - final true-query RMSE `0.00031491832864888683`;
  - accepted edits `17`;
  - candidates scored `61952`;
  - interpretation: JAX backend matches the useful CPU directed-group behavior
    on the known plateau-breaking probe.
- Adult DP high-throughput directed-group without positive fill:
  - output: `outputs/adult_qdte_directed_group_jax_50iter`;
  - candidates scored `78643200`;
  - accepted edits `400`;
  - final measured/unweighted loss `5.06997e+08`;
  - reason for low acceptance: one directed group per iteration with
    `directed_group_max_size=8`, so `accepted_per_iter=1024` was not actually
    reachable.
  - GPU0 active samples: average power `377.20W`, max power `418.05W`.
- Adult DP high-throughput directed-group with positive fill:
  - output: `outputs/adult_qdte_directed_group_jax_positive_fill_50iter`;
  - candidates scored `78643200`;
  - accepted edits `27223`;
  - final measured/unweighted loss `9.40652e+07`;
  - accepted edits per second `1078.28`;
  - candidates scored per second `3114989.26`;
  - GPU0 active samples: average power `345.46W`, max power `401.43W`.
- Adult DP high-throughput directed-group with positive fill plus augment:
  - output:
    `outputs/adult_qdte_directed_group_jax_positive_fill_augment_50iter`;
  - candidates scored `78643200`;
  - accepted edits `25131`;
  - final measured/unweighted loss `1.077e+08`;
  - transport time `46.28s` out of `69.04s` generation time;
  - accepted edits per second `364.00`;
  - GPU0 active samples: average power `146.45W`, max power `262.89W`.
- Adult DP 10-step no-augment verification after making augment optional:
  - output:
    `outputs/adult_qdte_directed_group_jax_positive_fill_noaugment_10iter`;
  - candidates scored `15728640`;
  - accepted edits `6458`;
  - final measured/unweighted loss `3.11243e+08`;
  - transport time `1.83s` and scoring time `9.23s`;
  - `directed_group_positive_fill_augment=false` in `runtime.json`.
- Adult DP high-throughput atom-flow 50-step comparison:
  - output: `outputs/adult_qdte_atom_flow_50iter`;
  - candidates scored `78643200`;
  - accepted edits `26785`;
  - final measured/unweighted loss `9.59141e+07`;
  - generation time `27.46s`, scoring time `22.59s`, transport time `4.80s`;
  - accepted edits per second `975.32`;
  - candidates scored per second `2863634.88`;
  - GPU0 active samples: average power `323.30W`, max power `369.60W`.
- New reusable high-power directed-group config:
  - `configs/adult_qdte_gpu_highpower_directed_group.yaml`;
  - same adult DP/workload/GPU-candidate settings as
    `configs/adult_qdte_gpu_highpower.yaml`;
  - transport switched to `directed_group`;
  - `directed_group_backend=jax`;
  - `directed_group_positive_fill=true`;
  - `directed_group_positive_fill_augment=false`.
- Adult DP high-throughput directed-group 500-step validation:
  - config: `configs/adult_qdte_gpu_highpower_directed_group.yaml`;
  - output: `outputs/adult_qdte_gpu_highpower_directed_group_500iter`;
  - power log: `outputs/gpu_power_directed_group_highpower_500iter.csv`;
  - true-query evaluation disabled for this performance run;
  - candidates scored `786432000`;
  - accepted edits `73371`;
  - initial measured/unweighted loss `5.43624e+08`;
  - final measured/unweighted loss `3.76538e+06`;
  - final RMS unweighted residual `29.5129`;
  - generation time `180.02s`, scoring time `170.66s`, transport time `8.99s`;
  - candidates scored per second `4368647.77`;
  - accepted edits per second `407.58`;
  - final incremental answer drift `0`;
  - GPU0 all samples: average power `387.37W`, max power `449.36W`;
  - GPU0 active samples: average power `420.45W`, max power `449.36W`;
  - samples with power >= `300W`: `338`, average power `427.85W`;
  - samples with power >= `400W`: `303`, average power `432.17W`.

Current Judgment:

- The `8` accepted edits per iteration result was not a candidate-generation
  failure. It was a transport bottleneck caused by using plateau-oriented
  directed-group selection as the only acceptance mechanism.
- For early-stage optimization where returned top-k edits are already positive,
  the correct high-throughput behavior is to accept a positive batch first.
  Directed-group search should mainly act as a fallback for plateau or
  compensating-edit situations.
- The GPU utilization goal is now met on the adult high-throughput path:
  effective active-run power is above `300W` on the 4090.
- Residual-updated directed-group augment is implemented but is not currently a
  good default for the high-throughput path. The dense top-k group search makes
  transport dominate runtime and drops effective GPU power below `300W`.
- Current recommended high-throughput setting:
  `directed_group_positive_fill=true` and
  `directed_group_positive_fill_augment=false`.
- On the 50-step adult DP comparison, directed-group positive-fill is slightly
  better than atom-flow under the same high-power candidate/scoring setup:
  - lower final loss: `9.40652e+07` vs `9.59141e+07`;
  - higher accepted edits/sec: `1078.28` vs `975.32`;
  - higher candidates/sec: `3114989.26` vs `2863634.88`;
  - higher active average GPU power: `345.46W` vs `323.30W`.
- The 500-step directed-group positive-fill run confirms the performance target:
  active average GPU power is `420.45W`, well above the requested `300W` floor
  and close to the `450W` card limit.
- Interpretation of the final 500-step loss scale:
  - final loss `3.76538e+06` is `0.5 * sum_q residual_q^2` over `8646`
    count-valued queries, not a percentage error;
  - equivalent RMS residual is `29.5129` counts/query;
  - relative RMS residual is `29.5129 / 45222 = 0.0006526`, about `0.065%`
    of the row count;
  - average measurement variance is `444.55`, so average raw noise std is
    about `21.08` counts; final RMS residual is about `1.40x` that average
    raw-noise std;
  - rough raw-noise loss scale `0.5 * num_queries * mean_variance` is
    `1.92178e+06`, so final loss is about `1.96x` this crude noise scale;
  - final loss is dominated by `mixed` queries: `68.55%` of total final loss,
    with mixed-family RMS residual `41.35` over `3019` queries.

Privacy Boundary:

- The adult runs above used `privacy.mode=dp` and did not enable true-query
  evaluation.
- Exact-target smoke probes are explicitly non-DP/oracle diagnostics and must
  not be used as DP paper results.

Next Recommended Task:

- Do not optimize the dense augment path further unless plateau diagnostics show
  it is needed. If augment becomes important, implement it as a sparse/fused
  top-k operation instead of re-evaluating dense candidate deltas.
- Use `configs/adult_qdte_gpu_highpower_directed_group.yaml` for the next longer
  high-power run. If 500-step behavior remains close to the 50-step result,
  promote directed-group positive-fill as the default high-throughput QDTE
  transport and keep atom-flow as a baseline/ablation.

## Non-DP Exact-Target Performance Limit And Similarity Probe - 2026-06-18

Task:

- Test QDTE's performance-limit behavior outside DP by using exact true
  workload answers as the target.
- Check whether a low-loss exact-target synthetic dataset simply copies the
  source dataset.

Setup:

- Base config:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`
- Common exact-target overrides:
  - `privacy.mode=oracle`;
  - `privacy.oracle_variance=1.0`;
  - `projection.project_partitions=false`;
  - `projection.clip_nonpartition=false`;
  - `projection.consistency.enabled=false`;
  - `projection.uncertainty.enabled=false`;
  - `qdte.objective_weighting=unweighted`.
- This is not DP. Exact true query answers are the optimization target.

Zero-Loss Sanity Check:

- Output:
  `outputs/exp_oracle_exact_real_init_reference`
- Initialized the synthetic table directly from the encoded real table:
  `outputs/exp_oracle_exact_real_init_reference/real_encoded.npy`.
- Result:
  - final measured/unweighted loss `0`;
  - true-query MAE `0`;
  - true-query RMSE `0`;
  - candidates scored `0`;
  - accepted edits `0`.
- Interpretation: zero loss is globally feasible in this exact-target,
  same-schema, same-row-count setting. Any nonzero result from QDTE is an
  optimization/search limitation, not an infeasibility floor.

QDTE Exact-Target Results:

- A-style QDTE, 2000 steps:
  - output `outputs/exp_oracle_exact_smoke_A_2000`;
  - initial loss `126267`;
  - final measured/unweighted loss `36`;
  - true-query MAE `0.00027272727272727263`;
  - true-query RMSE `0.0005454545454545454`;
  - max true-query error `0.0020000000000000018`;
  - candidates scored `512000`;
  - accepted edits `911`;
  - wall time `15.428s`.
- Wide A-style QDTE:
  - output `outputs/exp_oracle_exact_smoke_A_wide_10000`;
  - overrides included `total_candidates_per_iter=4096`,
    `num_active_targets=32`, `candidates_per_target=64`,
    `accepted_per_iter=16`, `max_iters=10000`, `stop_patience=2000`;
  - stopped at iter `3990` due patience;
  - final measured/unweighted loss `21`;
  - true-query MAE `0.00017355371900826434`;
  - true-query RMSE `0.0004165977904505306`;
  - max true-query error `0.0010000000000000009`;
  - candidates scored `16343040`;
  - accepted edits `849`;
  - wall time `1m28.272s`.
- Directed group probe from the wide result:
  - output `outputs/exp_oracle_exact_smoke_directed_group_probe`;
  - started from loss `21`;
  - final measured/unweighted loss `12`;
  - true-query MAE `9.917355371900827e-05`;
  - true-query RMSE `0.00031491832864888683`;
  - candidates scored `61952`;
  - accepted edits `17`;
  - wall time `3.760s`.
- Stronger directed group probe from the loss-12 result:
  - output `outputs/exp_oracle_exact_smoke_directed_group_probe2`;
  - started from loss `12`;
  - final measured/unweighted loss `10`;
  - true-query MAE `8.264462809917356e-05`;
  - true-query RMSE `0.0002874797872880345`;
  - candidates scored `415744`;
  - accepted edits `2`;
  - wall time `51.475s`.

Performance-Limit Interpretation:

- QDTE can drive exact-target loss very close to zero on the smoke workload:
  `126267 -> 36 -> 21 -> 12 -> 10`.
- It does not reach the known feasible zero-loss solution under the current
  search/transport stack.
- The observed plateau is not only a candidate-count issue:
  - increasing candidate pool improves `36 -> 21`;
  - group transport can break a single-edit/constructive-unit plateau and
    improve `21 -> 12 -> 10`;
  - later improvements become expensive and sparse.
- Current evidence: exact-target performance is limited by local search and
  grouped-edit discovery, not by DP noise or target inconsistency.

Similarity Diagnostic For The Loss-10 Synthetic Dataset:

- Compared:
  - real encoded table:
    `outputs/exp_oracle_exact_real_init_reference/real_encoded.npy`;
  - loss-10 synthetic table:
    `outputs/exp_oracle_exact_smoke_directed_group_probe2/synthetic_encoded.npy`.
- Diagnostic output:
  `outputs/exp_oracle_exact_smoke_directed_group_probe2/row_similarity_to_real.json`
- Results:
  - positional exact row matches: `0 / 1000`;
  - unique real rows: `190`;
  - unique synthetic rows: `394`;
  - unique row-set intersection: `141`;
  - unique row-set Jaccard: `0.3182844243792325`;
  - multiset exact row overlap: `424 / 1000`;
  - multiset Jaccard: `0.26903553299492383`;
  - row-count total variation fraction: `0.576`;
  - synthetic rows whose value pattern appears somewhere in real:
    `0.484`;
  - real rows whose value pattern appears somewhere in synthetic:
    `0.809`;
  - synthetic-to-real nearest Hamming distance histogram:
    `{0: 484, 1: 459, 2: 56, 3: 1, 4: 0, 5: 0}`;
  - real-to-synthetic nearest Hamming distance histogram:
    `{0: 809, 1: 191, 2: 0, 3: 0, 4: 0, 5: 0}`;
  - per-attribute marginal L1 differences:
    `[0, 0, 0, 0, 0]`.
- Column-permutation baseline:
  - independently permuting each real column gives multiset exact row overlap
    `128 / 1000`;
  - row-count TV fraction `0.872`;
  - nearest-Hamming mean `1.198`.

Similarity Interpretation:

- The loss-10 synthetic table is not a one-to-one copy of the source table:
  positional row matches are zero, row-count TV is high, and the synthetic table
  has many more unique row patterns than the real table.
- It still preserves substantial source-like structure:
  - `424` rows match real row patterns with multiplicity;
  - `484` synthetic rows have exact row patterns present in the real set;
  - nearly all remaining synthetic rows are Hamming-distance `1` or `2` from a
    real row.
- This is expected in non-DP exact-target fitting and should not be described as
  privacy-preserving. It is useful as a performance-limit probe, not as a
  release mechanism.

Changed Files:

- `docs/HANDOFF.md`

New Output Artifacts:

- `outputs/exp_oracle_exact_real_init_reference/`
- `outputs/exp_oracle_exact_smoke_A_2000/`
- `outputs/exp_oracle_exact_smoke_A_wide_10000/`
- `outputs/exp_oracle_exact_smoke_directed_group_probe/`
- `outputs/exp_oracle_exact_smoke_directed_group_probe2/`
- `outputs/exp_oracle_exact_smoke_directed_group_probe2/row_similarity_to_real.json`

Tests And Probes Run:

- Oracle exact-target zero-loss sanity check.
- Oracle exact-target A-style 2000-step run.
- Oracle exact-target wide A-style run.
- Directed-group continuation probes from loss `21` and loss `12`.
- Row-level similarity analysis between the loss-10 synthetic table and the
  encoded real table.
- One larger directed-group run was manually interrupted because the
  Python/NumPy group search was too slow for an exploratory probe.
- `git diff --check`
  - Result: passed.

Current Status:

- Exact-target QDTE has a known zero feasible optimum on this smoke workload.
- Current best generated synthetic loss is `10`, not zero.
- The best synthetic is not identical to the real table, but it remains strongly
  source-like at the row-pattern/nearest-neighbor level.

Next Recommended Task:

- If exact-target performance-limit experiments remain important, implement a
  faster grouped-edit or block-repair search before running larger workloads.
  The current directed-group path is useful diagnostically but too CPU-heavy for
  full sweeps.

## QDTE Outside Differential Privacy - 2026-06-18

Question:

- Does QDTE still make sense outside the differential-privacy setting?
- If the target queries are exact true answers instead of noisy/projected DP
  measurements, can QDTE reach the theoretical zero-loss limit?

Current Judgment:

- QDTE is not inherently DP-specific. The DP setting supplies a noisy/projected
  target and variances, but the core algorithm is a query-targeted synthetic
  dataset compiler:
  - maintain `residual = target - answer_syn`;
  - generate directed record edits;
  - score edits by full-workload edit advantage;
  - apply accepted edits to move one synthetic dataset toward the query target.
- Outside DP, the natural target is the exact query answer vector from the real
  dataset. Then the objective can be viewed as exact workload matching.

When Zero Loss Is Theoretically Achievable:

- If:
  - the synthetic dataset has the same number of records as the real dataset;
  - the synthetic schema/domain matches the real encoded schema;
  - the workload answers are exact counts or rates representable by that record
    count;
  - the target query vector is computed from an actual dataset;
  - the optimization objective uses that exact target directly;
- then a zero-loss solution exists: the real dataset itself is feasible and
  exactly matches every query in the workload.
- This does not mean QDTE will automatically find it; it only means the global
  optimum is zero.

When Zero Loss May Not Be Achievable:

- If `N_syn != N_real`, exact target rates may not be representable by
  `N_syn` integer records. The lower bound is then a quantization/integrality
  floor.
- If the synthetic schema is coarser than the source schema, some exact query
  answers may be impossible to realize.
- If the target vector is inconsistent, for example after arbitrary postprocess
  constraints or incompatible marginals, no dataset may match all targets.
- If the objective includes weights, clipping, projection, or transformed
  targets, the zero-loss statement applies only to the chosen transformed
  target, not necessarily to original exact answers.
- If the workload includes queries outside the representable query evaluator,
  exact matching is impossible in the current implementation.

Algorithmic Guarantee:

- The current QDTE implementation is greedy/local-search style. It does not
  prove global convergence to zero, even when a zero-loss dataset exists.
- Practical convergence depends on:
  - candidate proposal coverage;
  - whether the edit generator can reach the needed records/values;
  - transport/acceptance policy;
  - iteration budget;
  - local optima and neutral moves;
  - whether multi-restart or population search is used.
- Therefore the right non-DP experiment is not just "is zero feasible?", but
  "how close does QDTE get to zero compared with stronger non-DP fitting
  baselines under matched compute?"

Why It May Still Be Useful Without DP:

- If privacy is irrelevant and the full source dataset can be released, exact
  generation is unnecessary because releasing the real data already gives zero
  workload error.
- QDTE is useful outside DP when the goal is not release of the original data,
  but query-targeted compilation:
  - compress a large dataset into a smaller synthetic dataset;
  - fit public aggregate constraints into a record-level synthetic table;
  - generate benchmark data matching selected marginals/ranges;
  - initialize or refine synthetic data for downstream simulations;
  - study directed local edit search independent of DP noise.
- In this framing, QDTE's research value is a directed dataset compiler, while
  DP is an important application where the target is noisy/projected and the
  optimization boundary is privacy constrained.

Experiment Recommendation:

- Add a non-DP exact-target mode or config:
  - `privacy.mode=none` or equivalent;
  - target is exact true workload answers;
  - objective weighting should likely start as unweighted;
  - set `N_syn=N_real` for the first zero-loss feasibility probe.
- Run:
  - single-layer A/QDTE with exact targets;
  - random mutation + edit advantage;
  - private-GSD-like population baseline without DP noise;
  - optionally an exact solver or IPF/LP baseline for small domains.
- Report:
  - final exact workload loss;
  - whether it reaches numerical zero;
  - accepted edits and runtime;
  - distance to zero under different `N_syn` values.

Changed Files:

- `docs/HANDOFF.md`

Tests And Probes Run:

- Documentation-only discussion update.
- `git diff --check`
  - Result: passed.

Current Status:

- The current judgment is that QDTE has a meaningful non-DP interpretation as a
  query-targeted dataset compiler.
- Zero loss is globally feasible under same-size, same-domain, exact-target
  conditions, but current QDTE does not guarantee finding the zero-loss dataset.

Next Recommended Task:

- Implement or verify a clean exact-target non-DP config and run a small
  zero-loss feasibility probe before claiming non-DP convergence behavior.

## Randomness Semantics For Population Continuation - 2026-06-18

Question:

- Why did the first 10 generations of the 100-generation random-row population
  run exactly match the earlier 10-generation random-row run?

Answer:

- The match is expected because the 100-generation run was launched as a
  same-seed/same-measurement continuation, not as an independent random
  replicate.
- Population restart seeds are deterministic:
  - the base seed is `run.seed`;
  - each individual gets `seed = base_seed + index * population.seed_stride`;
  - with `run.seed=0` and `seed_stride=1000`, the first generation uses
    `0, 1000, 2000, ...`.
- Each QDTE worker creates `np.random.default_rng(run.seed)`, so the same
  individual seed gives the same synthetic initialization and same candidate
  proposal stream.
- `independent_oneway` initialization is random, but it samples from that
  seeded RNG; therefore it is reproducible.
- Crossover randomness is also deterministic:
  `np.random.default_rng(base_seed + 7919)` drives parent selection and row
  crossover. With the same previous-generation ranking and the same seed, it
  chooses the same parent pairs and row swaps.
- The 100-generation run also reused the same DP measurement artifact as the
  10-generation run. Therefore the target, residual landscape, initial
  generation seeds, and crossover stream were identical for the first 10
  generations.

Credibility Boundary:

- The 100-generation result is credible as an extension of the previous
  trajectory.
- It is not credible as an independent replicate or evidence of cross-seed
  robustness.
- To test algorithmic randomness independently while controlling DP noise, run
  the same experiment with a new `run.seed` but the same
  `measurement.reuse_from`.
- To test total pipeline randomness, run with a new `run.seed` and a fresh DP
  measurement.

Changed Files:

- `docs/HANDOFF.md`

Tests And Probes Run:

- Inspected:
  - `scripts/run_population.py`;
  - `qdte/evolution/engine.py`;
  - `qdte/evolution/initialization.py`;
  - `qdte/queries/workload.py`.

Current Status:

- The earlier 100-generation experiment should be labeled explicitly as
  same-seed/same-measurement continuation.

Next Recommended Task:

- If population-level evolution remains of interest, run at least one
  `run.seed=1` replicate reusing the same measurement artifact to separate
  algorithmic randomness from DP-noise randomness.

## A-Style Random-Row Outer Population 100-Generation Continuation - 2026-06-18

Task:

- Extend the A-style inner + random-row outer population experiment from `10`
  generations to `100` generations.
- Requested settings:
  - `population.size=48`;
  - `population.inner_iters=100`;
  - random-row crossover;
  - `population.generations=100`.

Important Credibility Note:

- This run intentionally reused the previous 10-generation DP measurement
  artifact:
  `outputs/pop48_gen10_A_orthogonal_random_row/measurement`.
- It also kept the same base `run.seed=0` and `population.seed_stride=1000`.
- Therefore the first 10 generations are expected to reproduce the previous
  10-generation run exactly. This run is a same-seed/same-measurement
  continuation, not an independent random replicate.
- Credible interpretation:
  - answers whether the same random trajectory improves when extended to 100
    generations;
  - preserves the DP boundary because optimization still uses only the reused
    noisy/projected measurements.
- Not credible interpretation:
  - does not prove random-row outer evolution is robust across seeds;
  - should not be counted as a multi-seed ablation.

Experiment:

- Output:
  `outputs/pop48_gen100_A_orthogonal_random_row`
- Base config:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`
- Important overrides:
  - `measurement.reuse_from=outputs/pop48_gen10_A_orthogonal_random_row/measurement`;
  - `population.generations=100`;
  - `population.size=48`;
  - `population.elite_count=4`;
  - `population.inner_iters=100`;
  - `population.evaluate_individuals=false`;
  - `population.parallel.enabled=true`;
  - `population.parallel.gpu_devices=0,1`;
  - `population.parallel.workers_per_gpu=24`;
  - `population.crossover.enabled=true`;
  - `population.crossover.mode=random_row`;
  - `population.crossover.children=44`;
  - `population.crossover.parent_pool=16`;
  - `population.crossover.fraction=0.5`;
  - `evaluation.compute_true_query_error=false` during optimization.

Result:

- Wall time: `8m02.911s`.
- Peak sampled GPU memory:
  - GPU0 `11204 MiB`;
  - GPU1 `10965 MiB`.
- Output size: `1.2G`.
- Best measured/unweighted loss: `16.92483773235196`.
- Best appeared at generation `70`.
- Best individual:
  - kind `crossover`;
  - slot `10`;
  - index `3370`;
  - seed `3370000`;
  - output
    `outputs/pop48_gen100_A_orthogonal_random_row/generation_0070/individual_010`;
  - random-row crossover swapped `519` rows;
  - subsequent inner QDTE scored `25600` candidates and accepted `154` edits.

Generation Checkpoints:

- gen `0`: global best `177.1397391796481`, accepted edits `38019`;
- gen `1`: global best `22.3917516708743`, accepted edits `11310`;
- gen `2`: global best `19.34641327861661`, accepted edits `8840`;
- gen `6`: global best `17.701817417181672`, accepted edits `8428`;
- gen `10`: still `17.701817417181672`, accepted edits `8015`;
- gen `20`: global best `17.626083517111482`, accepted edits `7540`;
- gen `21`: global best `17.225997829474153`, accepted edits `7364`;
- gen `65`: global best `17.147170925177278`, accepted edits `7331`;
- gen `70`: global best `16.92483773235196`, accepted edits `7242`;
- gen `99`: still `16.92483773235196`, accepted edits `7484`.

Offline True-Query Evaluation:

- Evaluation output:
  `outputs/eval_pop48_A_orthogonal_random_row_gen100_best_true_rmse`
- Evaluation reused:
  `outputs/pop48_gen100_A_orthogonal_random_row/measurement`.
- Evaluation loaded:
  `outputs/pop48_gen100_A_orthogonal_random_row/generation_0070/individual_010/synthetic_encoded.npy`.
- Evaluation set `qdte.max_iters=0`.
- The evaluation log confirms `Candidates scored: 0` and
  `Accepted edits: 0`.
- Result:
  - measured/unweighted loss `16.92483773235196`;
  - true-query MAE `0.0018305785123966943`;
  - true-query RMSE `0.0023275212935440198`;
  - true-query max error `0.006999999999999999`.
- Hash checks:
  - source best synthetic and evaluation synthetic hashes match:
    `afe3d29a4975ad4b29f4278e591f285ae19d6c5665dd7be94c4530bb9079c410`;
  - source measurement `queries.json` and evaluation `queries.json` hashes
    match:
    `9728cdce2c813b9a6a46f72a3083eba56d1097056dece460777056959daffb25`.

Comparison:

- Single-layer A:
  - measured/unweighted loss `21.849774742163362`;
  - true-query MAE `0.0017644628099173555`;
  - true-query RMSE `0.0022790820248584868`.
- A + random-row outer, 10-generation same-seed run:
  - measured/unweighted loss `17.701817417181672`;
  - true-query MAE `0.0018057851239669425`;
  - true-query RMSE `0.0023096992756204545`.
- A + random-row outer, 100-generation same-seed continuation:
  - measured/unweighted loss `16.92483773235196`;
  - true-query MAE `0.0018305785123966943`;
  - true-query RMSE `0.0023275212935440198`.
- A + context-aware outer, 100-generation run:
  - measured/unweighted loss `17.358035469092073`;
  - true-query MAE `0.001822314049586777`;
  - true-query RMSE `0.0023363813228902543`.

Interpretation:

- Extending the same random-row trajectory from 10 to 100 generations reduced
  measured/projected loss from `17.701817417181672` to
  `16.92483773235196`.
- Offline true-query RMSE got worse over that same extension:
  `0.0023096992756204545` to `0.0023275212935440198`.
- The result supports the current concern: the outer population loop is
  continuing to fit the noisy/projected target, but this is not translating into
  better true-query accuracy on this workload.
- Unless held-out or multi-seed experiments show a true-error gain, the outer
  population loop should stay out of the main algorithm narrative and be kept as
  an ablation/negative result.

Changed Files:

- `docs/HANDOFF.md`

Tests And Probes Run:

- A-style random-row 100-generation same-seed continuation as described above.
- Offline `qdte.max_iters=0` true-query evaluation of the best generation-70
  synthetic table.
- `jq` inspection of:
  - `outputs/pop48_gen100_A_orthogonal_random_row/population_summary.json`;
  - `outputs/eval_pop48_A_orthogonal_random_row_gen100_best_true_rmse/metrics_final.json`;
  - reference `metrics_final.json` files for single-layer A, random-row 10-gen,
    and context-aware population.
- `sha256sum` checks for source/evaluation synthetic and query catalogues.
- `git diff --check`
  - Result: passed.

Current Status:

- The requested 100-generation random-row outer population run is complete.
- The experiment is reliable as a same-seed/same-measurement continuation.
- It is not an independent random replicate.
- The observed pattern is lower measured loss but worse true-query RMSE.

Next Recommended Task:

- Treat the population layer as a candidate ablation rather than the main QDTE
  story unless independent seeds or a held-out/DP-valid validation workload show
  a real true-error improvement.
- If further checking is desired, run `run.seed=1` and `run.seed=2` with the
  same reused measurement to isolate algorithmic randomness from DP-noise
  randomness.

## A-Style Inner With Random-Row Outer Population - 2026-06-18

Task:

- Run the best current A-style single-layer QDTE method inside the outer
  population loop with random row crossover.
- Requested settings:
  - `population.generations=10`;
  - `population.size=48`;
  - `population.inner_iters=100`;
  - random crossover.

Environment Note:

- Installed system `jq` for JSON result inspection:
  `sudo -n apt-get update && sudo -n apt-get install -y jq`.

Experiment:

- Output:
  `outputs/pop48_gen10_A_orthogonal_random_row`
- Base config:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`
- Important overrides:
  - `population.generations=10`;
  - `population.size=48`;
  - `population.elite_count=4`;
  - `population.inner_iters=100`;
  - `population.crossover.enabled=true`;
  - `population.crossover.mode=random_row`;
  - `population.crossover.children=44`;
  - `population.crossover.parent_pool=16`;
  - `population.crossover.fraction=0.5`;
  - `population.parallel.enabled=true`;
  - `population.parallel.gpu_devices=0,1`;
  - `population.parallel.workers_per_gpu=24`;
  - true-query evaluation disabled during optimization for speed.

Result:

- Wall time: `real 92.58s`.
- Peak sampled GPU memory:
  - GPU0 `11204 MiB`;
  - GPU1 `10965 MiB`.
- Output size: `116M`.
- Best measured/unweighted loss: `17.701817417181672`.
- Best appeared at generation `6`.
- Best individual:
  - kind `crossover`;
  - slot `29`;
  - index `317`;
  - seed `317000`;
  - output
    `outputs/pop48_gen10_A_orthogonal_random_row/generation_0006/individual_029`;
  - parent indices `[242, 249]`;
  - random-row crossover swapped `495` rows;
  - subsequent inner QDTE scored `25600` candidates and accepted `197` edits.

Generation Checkpoints:

- gen `0`: best/global best `177.1397391796481`, accepted edits `38019`;
- gen `1`: best/global best `22.3917516708743`, accepted edits `11310`;
- gen `2`: best/global best `19.34641327861661`, accepted edits `8840`;
- gen `3`: best/global best `18.47868957523221`, accepted edits `8775`;
- gen `5`: best/global best `18.183885479010286`, accepted edits `8462`;
- gen `6`: best/global best `17.701817417181672`, accepted edits `8428`;
- gen `9`: still `17.701817417181672`, accepted edits `8165`.

Offline True-Query Evaluation:

- Evaluation output:
  `outputs/eval_pop48_A_orthogonal_random_row_best_true_rmse`
- Evaluation reused the random-row population measurement artifact, loaded the
  generation-6 best synthetic table, and set `qdte.max_iters=0`.
- Exact true answers were used only for offline evaluation.
- Evaluation result:
  - measured/unweighted loss `17.701817417181672`;
  - true-query MAE `0.0018057851239669425`;
  - true-query RMSE `0.0023096992756204545`;
  - true-query max error `0.006999999999999999`;
  - candidates scored `0`;
  - accepted edits `0`.

Comparison Against Current Reference Runs:

- Single-layer A:
  - output
    `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair`;
  - measured/unweighted loss `21.849774742163362`;
  - true-query MAE `0.0017644628099173555`;
  - true-query RMSE `0.0022790820248584868`.
- A-style outer loop with context-aware crossover:
  - output `outputs/pop48_gen100_A_orthogonal_context_aware`;
  - measured/unweighted loss `17.358035469092073`;
  - offline true-query MAE `0.001822314049586777`;
  - offline true-query RMSE `0.0023363813228902543`;
  - best crossover accepted `0` context-aware donor edits, so the winning path
    mostly behaved like a clone plus short inner continuation.
- A-style outer loop with random-row crossover:
  - output `outputs/pop48_gen10_A_orthogonal_random_row`;
  - measured/unweighted loss `17.701817417181672`;
  - offline true-query MAE `0.0018057851239669425`;
  - offline true-query RMSE `0.0023096992756204545`;
  - winning crossover actually swapped rows (`495`) and then improved through
    `197` accepted inner QDTE edits.

Interpretation:

- Random-row outer evolution improves the measured/projected objective relative
  to the single-layer A run.
- It does not improve offline exact-workload true RMSE over single-layer A in
  this run. It is slightly better than the context-aware population run on true
  RMSE, while slightly worse on measured loss.
- The current evidence still suggests that the outer loop can fit the
  noisy/projected target more strongly, but this does not automatically
  translate into lower true-query RMSE.
- If the paper keeps the outer loop, it should be framed as an optional
  exploration layer unless a held-out or multi-seed experiment shows a real
  generalization gain.

Changed Files:

- `docs/HANDOFF.md`

Tests And Probes Run:

- `jq` inspection of
  `outputs/pop48_gen10_A_orthogonal_random_row/population_summary.json`.
- `jq` comparison of `metrics_final.json` for:
  - `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair`;
  - `outputs/eval_pop48_A_orthogonal_context_best_true_rmse`;
  - `outputs/eval_pop48_A_orthogonal_random_row_best_true_rmse`.
- Parsed `outputs/pop48_gen10_A_orthogonal_random_row/gpu_mem.csv` to confirm
  peak sampled GPU memory.
- `git diff --check`
  - Result: passed.

Current Status:

- The requested A + random-row outer population run is complete.
- The run result is reliable for measured/projected objective comparison.
- The true-query RMSE number is reliable only as offline exact-workload
  evaluation of a fixed synthetic table, not as a DP-safe optimization signal.

Next Recommended Task:

- Before investing further in the population layer, run a multi-seed comparison
  of single-layer A, A + random-row population, and A + context-aware population
  with a held-out workload. The decision criterion should include held-out
  true-query MAE/RMSE or a DP-valid validation workload, not measured loss alone.

## Best A-Style Inner With Outer Population Loop - 2026-06-18

Task:

- Verify whether the outer population loop is still weak when applied to the
  best current single-layer QDTE method.
- The motivating concern was that the previous population smoke used the weak
  default `configs/smoke.yaml` inner settings, not the best A-style inner
  method.

Best Single-Layer Baseline:

- Baseline output:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair`
- Config:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`
- Key settings:
  - orthogonal mixed workload;
  - `projection.consistency.method=local_table_feasible_jax`;
  - `qdte.transport_mode=constructive_pair`;
  - `qdte.objective_weighting=unweighted`;
  - `qdte.max_iters=2000`;
  - `qdte.stop_patience=2000`.
- Single-layer result:
  - final measured/unweighted loss `21.849774742163362`;
  - true query RMSE `0.0022790820248584868`;
  - candidates scored `512000`;
  - accepted edits `955`.

Population Experiment:

- Output:
  `outputs/pop48_gen100_A_orthogonal_context_aware`
- Base config:
  `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/config_resolved.yaml`
- Overrides:
  - `population.generations=100`;
  - `population.size=48`;
  - `population.elite_count=4`;
  - `population.inner_iters=100`;
  - `population.crossover.enabled=true`;
  - `population.crossover.mode=context_aware`;
  - `population.crossover.children=44`;
  - `population.crossover.parent_pool=16`;
  - `population.crossover.candidates=512`;
  - `population.parallel.enabled=true`;
  - `population.parallel.gpu_devices=0,1`;
  - `population.parallel.workers_per_gpu=24`;
  - true-query evaluation disabled for speed.

Results:

- Final best measured/unweighted loss: `17.358035469092073`.
- Best appeared at generation `2`.
- Wall time: `real 477.92s`.
- Peak sampled GPU memory:
  - GPU0 `11204 MiB`;
  - GPU1 `10965 MiB`.
- Output size: `1.3G`.

Offline True-Query Evaluation:

- The population run disabled `evaluation.compute_true_query_error` for speed,
  so RMSE was computed afterward as offline evaluation only.
- Evaluation output:
  `outputs/eval_pop48_A_orthogonal_context_best_true_rmse`
- Evaluation command reused the same measurement artifact, loaded the best
  synthetic table from generation `2`, and set `qdte.max_iters=0`.
- Comparison:
  - single-layer A measured/unweighted loss:
    `21.849774742163362`;
  - population best measured/unweighted loss:
    `17.358035469092073`;
  - single-layer A true-query MAE:
    `0.0017644628099173555`;
  - population best true-query MAE:
    `0.001822314049586777`;
  - single-layer A true-query RMSE:
    `0.0022790820248584868`;
  - population best true-query RMSE:
    `0.0023363813228902543`;
  - both had true-query max error `0.006999999999999999`.
- Interpretation:
  - the outer population loop improved measured/projected objective;
  - it did not improve offline true-query RMSE in this run;
  - this is consistent with slightly stronger fitting to the noisy/projected
    target rather than guaranteed improvement of the real query answers.

Reliability Check:

- `outputs/eval_pop48_A_orthogonal_context_best_true_rmse/config_resolved.yaml`
  has `qdte.max_iters=0`, `evaluation.compute_true_query_error=true`, and
  `measurement.reuse_from=outputs/pop48_gen100_A_orthogonal_context_aware/measurement`.
- The evaluation log confirms:
  - initialized from
    `outputs/pop48_gen100_A_orthogonal_context_aware/generation_0002/individual_038/synthetic_encoded.npy`;
  - `Candidates scored: 0`;
  - `Accepted edits: 0`;
  - exact true query answers were computed only for offline evaluation.
- Hash checks:
  - source best synthetic and evaluation synthetic hashes match:
    `a85eb43d8eb7dfd6cea92b7815bce7af70e91fb12e909c2e9e5a927ea6880a15`;
  - source measurement `queries.json` and evaluation `queries.json` hashes
    match:
    `9728cdce2c813b9a6a46f72a3083eba56d1097056dece460777056959daffb25`.
- Therefore this RMSE is reliable as an offline exact-workload evaluation of
  that fixed synthetic table. It is not a DP-safe optimization signal and it is
  not a held-out-workload RMSE.

Generation Checkpoints:

- gen `0`: best/global best `177.1397391796481`;
- gen `1`: best/global best `18.719519996679963`;
- gen `2`: best/global best `17.358035469092073`;
- gen `3`: unchanged, generation accepted edits `77`;
- gen `5`: unchanged, generation accepted edits `0`;
- gen `10`: unchanged;
- gen `20`: unchanged;
- gen `50`: unchanged;
- gen `99`: unchanged.

Important Interpretation:

- This corrects the previous conclusion:
  - the outer loop is not intrinsically worse;
  - when attached to the best A-style inner method, it improves measured loss
    from `21.849774742163362` to `17.358035469092073`.
- However, the useful improvement happens almost entirely in the first two
  generations.
- After generation `5`, the population has converged to a zero-positive-edit
  plateau and the remaining generations are mostly wasted compute.
- The best individual is labeled `crossover`, but the context-aware crossover
  accepted `0` donor rows:
  - parent losses were `18.719519996679963` and `22.689061069525422`;
  - the child reached `17.358035469092073` through `8` subsequent inner QDTE
    edits.
- Therefore this result supports the outer loop as parallel multi-start /
  clone-and-continue over a strong inner method, not yet as evidence that
  donor-row recombination itself is the key mechanism.

Current Judgment:

- The earlier poor population result was mostly a configuration mismatch.
- The strongest story remains:
  1. single-layer QDTE / A-style directed edit evolution is the core algorithm;
  2. outer population evolution can improve it as an optional parallel search
     layer;
  3. the current outer layer needs stagnation-aware early stopping / restart
     allocation because 100 generations are unnecessary here.
- The paper can mention the outer layer as a scalable wrapper or ablation, but
  it should not replace the single-layer directed edit narrative.

Next Recommended Task:

- Add stagnation-aware population control:
  - stop generations once global best has not improved for a configurable
    patience;
  - reallocate zero-positive-edit clones to fresh restarts or high-temperature
    mutations;
  - keep compact output only for elites and checkpoints.
- Optionally run `random_row` with the same A-style inner settings if we want to
  separate "population size" from "context-aware safety" in the final ablation.

## Population Loop Interpretation And Failure Analysis - 2026-06-18

Question:

- The outer population loop appears to add little value.
- Analyze why the two-level loop is not obviously better than the single-layer
  QDTE algorithm.
- Decide whether the paper still needs an outer-loop narrative.

Key Clarification:

- The latest 48-population smoke experiments were not run with the strongest
  single-layer A configuration.
- They used `configs/smoke.yaml`, whose inner QDTE path is the default
  `microbatch_greedy` style setting.
- Existing single-layer A-style smoke results are much stronger under different
  inner settings:
  - `outputs/exp_unweightedobj_original_consistency2000_constructive_pair`:
    final measured/unweighted loss `56.984528037698944`;
  - `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair`:
    final measured/unweighted loss `21.849774742163362`;
  - `outputs/exp_constructiveA2000_constructive_pair`:
    final measured loss `2.2138641508483943` under its older weighted objective
    setting.
- Therefore the population smoke losses around `1431` are not comparable to the
  strongest single-layer A results. This is a configuration mismatch, not proof
  that the outer loop is intrinsically worse.

Same-Config Check:

- A continuous single-chain run on the same `configs/smoke.yaml` objective:
  - output: `outputs/single_smoke_10000_continuous`;
  - command used `qdte.max_iters=10000` and `qdte.stop_patience=100`;
  - stopped at iteration `1414` after no positive edits;
  - final measured/unweighted loss `1470.3388388952399`;
  - candidates scored `361984`;
  - accepted edits `1794`.
- Under this same weak inner setting, population did help measured loss:
  - 48-pop `random_row`: `1447.1932299932623`;
  - 48-pop `context_aware`: `1431.6937743028784`.
- However, the improvement is very inefficient relative to the much larger
  population budget.

Why The Outer Loop Is Weak Right Now:

- The current outer loop mostly redistributes the same inner local search:
  it does not add a fundamentally new edit generator.
- The best context-aware crossover child accepted `0` donor rows during the
  crossover construction; its improvement came from `3` subsequent inner QDTE
  edits. That means the best "crossover" event was effectively clone-and-continue.
- Context-aware crossover becomes too conservative near a plateau:
  once donor replacements have no positive measured edit advantage, it imports
  nothing.
- Random-row crossover explores more aggressively, but it is mostly destructive
  and requires the inner loop to repair the damage.
- After generation `0`, the current 48-pop setup uses `4` elite clones plus `44`
  crossover children, leaving no fresh random restart slots. Diversity can
  collapse into variants of the same few parents.
- The worker budget is fixed per individual per generation. Once individuals
  hit a zero-positive-edit plateau, many workers spend full budgets confirming
  no local edit exists instead of being reassigned to more exploratory states.
- Each generation resets QDTE's local scheduling/debt state. This can make the
  loop less coherent than one continuous optimizer when long-horizon query
  pressure matters.

Current Judgment:

- The outer loop should not be the core paper narrative yet.
- The defensible core remains the single-dataset, individual-edit-level
  directed evolution / edit-advantage method.
- The population loop is currently best described as an optional engineering
  layer for parallel multi-start, robustness, and search-space probing.
- It becomes a paper-level contribution only if it improves the strongest
  single-layer A configuration under equal or clearly reported compute budgets.

Recommended Next Experiments:

- Re-run population with the actual A-style inner settings, not default smoke
  microbatch settings.
- Compare:
  - single-layer A with the same total candidate budget;
  - A + random restarts;
  - A + context-aware safe donor import;
  - A + random-row crossover.
- Add a stagnation-aware outer policy:
  - keep global elites;
  - reserve some slots for fresh restarts or high-temperature random mutation;
  - stop spending full inner budgets on clones with zero positive edits;
  - allocate more workers to genuinely improving or diverse states.
- Add compact output retention before large reruns.

Changed Files:

- `docs/HANDOFF.md`

Tests / Checks Run:

- Parsed existing population and single-layer metrics.
- Ran same-config continuous single-chain smoke:
  `outputs/single_smoke_10000_continuous`.
- Ran offline true-query evaluation for the A-style population best:
  `outputs/eval_pop48_A_orthogonal_context_best_true_rmse`.

Next Recommended Task:

- Decide whether to pause the outer-loop line as optional engineering, or run
  one A-style population experiment to test whether the loop still helps when
  the inner algorithm is the best known QDTE variant.

## Population-48 Random vs Context-Aware Crossover Smoke - 2026-06-18

Task:

- Compare `random_row` crossover and `context_aware` crossover with large
  parallel population search.
- Use `48` concurrently evolving datasets, inner QDTE budget `100`, and outer
  generation count `100`.

Shared Setup:

- Config: `configs/smoke.yaml`.
- `population.size=48`.
- `population.elite_count=4`.
- `population.generations=100`.
- `population.inner_iters=100`.
- `qdte.stop_patience=100` so the inner budget is not cut short by the smoke
  config's default patience `15`.
- `population.crossover.enabled=true`.
- `population.crossover.children=44`.
- `population.crossover.parent_pool=16`.
- `population.crossover.fraction=0.5`.
- `population.crossover.candidates=512`.
- `population.crossover.max_edits=0`, so context-aware max edits is inferred
  from the crossover fraction.
- `population.parallel.enabled=true`.
- `population.parallel.gpu_devices=0,1`.
- `population.parallel.workers_per_gpu=24`.
- Evaluation true-query metrics and synthetic CSV output disabled.

Results:

- `random_row`:
  - output: `outputs/pop48_gen100_random_row_smoke`;
  - wall time: `real 170.84s`;
  - peak sampled GPU memory: GPU0 `11094 MiB`, GPU1 `10965 MiB`;
  - best measured/unweighted loss: `1447.1932299932623`;
  - best generation: `69`;
  - output size: `1.1G`.
- `context_aware`:
  - output: `outputs/pop48_gen100_context_aware_smoke`;
  - wall time: `real 156.51s`;
  - peak sampled GPU memory: GPU0 `11190 MiB`, GPU1 `10965 MiB`;
  - best measured/unweighted loss: `1431.6937743028784`;
  - best generation: `16`;
  - output size: `1.3G`.

Selected Generation Checkpoints:

- `random_row` global best:
  - gen `0`: `3712.289897860685`;
  - gen `1`: `1775.3290577730322`;
  - gen `2`: `1538.6653405031348`;
  - gen `5`: `1469.4965012869025`;
  - gen `10`: `1454.4163795313025`;
  - gen `20`: `1450.6496506532812`;
  - gen `50`: `1449.5913811525488`;
  - gen `69`: `1447.1932299932623`;
  - gen `99`: `1447.1932299932623`.
- `context_aware` global best:
  - gen `0`: `3712.289897860685`;
  - gen `1`: `1756.7928494826942`;
  - gen `2`: `1527.5099216303015`;
  - gen `5`: `1457.1753493150854`;
  - gen `10`: `1439.7126308282996`;
  - gen `16`: `1431.6937743028784`;
  - gen `20`: `1431.6937743028784`;
  - gen `50`: `1431.6937743028784`;
  - gen `99`: `1431.6937743028784`.

Interpretation:

- Under this 48-population smoke budget, `context_aware` beat `random_row` by
  `15.4994556903839` measured/unweighted loss units.
- `context_aware` also reached its best state much earlier, generation `16`
  instead of generation `69`.
- The best context-aware individual was labeled `crossover`, but its actual
  crossover step accepted `0` donor rows:
  - parent losses were `1434.2934223016882` and `1434.048078139701`;
  - the child then improved to `1431.6937743028784` through `3` inner QDTE
    edits.
- Therefore the current win should be interpreted carefully:
  - context-aware crossover is acting as a conservative gate that avoids
    harmful donor-row imports;
  - many "crossover" children are effectively copied from a strong parent and
    then get a fresh inner-QDTE local search seed;
  - this still supports the usefulness of measured-objective gated population
    evolution, but it does not yet prove that donor-row recombination itself is
    the source of the gain.

Current Judgment:

- Large parallel population does help the outer loop compared with the earlier
  size-2 smoke (`1460.0758429369116` best), because both 48-population variants
  found better states.
- `context_aware` is currently the better crossover mode for this smoke setting.
- The next algorithmic question is whether to explicitly formalize this as
  "safe donor import / clone-and-continue when no positive donor edit exists",
  rather than presenting it as ordinary random crossover.

Changed Files:

- `docs/HANDOFF.md`

Tests / Checks Run:

- `random_row` 48-population, 100-generation smoke as described above.
- `context_aware` 48-population, 100-generation smoke as described above.

Next Recommended Task:

- Add a compact output-retention option before repeating this on larger
  workloads; the two smoke runs wrote about `2.4G` combined.
- Run a smaller Adult highpower version using the measured safe capacity:
  `population.size=6`, `workers_per_gpu=3`, or aggressive
  `population.size=8`, `workers_per_gpu=4`.

## Persistent GPU-Parallel Population Workers - 2026-06-18

Task:

- Change the multi-dataset population runner from serial execution to GPU
  parallel execution.
- Avoid making the two-level evolution story slower than single-dataset QDTE.

Implementation:

- `scripts/run_population.py` now supports `population.parallel`:
  - `population.parallel.enabled`;
  - `population.parallel.gpu_devices`, either `auto`, a comma string such as
    `0,1`, or a list;
  - `population.parallel.workers_per_gpu`;
  - `population.parallel.workers`, where `0` means auto from GPU slots.
- Parallel execution uses a persistent GPU worker pool:
  - one long-lived worker process is started per GPU slot;
  - each worker is bound with `CUDA_VISIBLE_DEVICES`;
  - tasks are sent over stdin as config paths;
  - QDTE stdout/stderr for each task is redirected to that task's
    `worker_stdout.log`;
  - the worker's protocol stdout stays clean and returns task status;
  - JAX is imported once per worker and reused across generations.
- Added `scripts/population_worker.py` as the persistent worker entrypoint.
- The population runner still preserves the DP boundary:
  - measurement/projection is done once;
  - worker tasks use `measurement.reuse_from`;
  - parent selection and crossover are driven by measured objective and
    projected/noisy target only.

Why Persistent Workers:

- A naive subprocess-per-individual parallel runner works functionally, but
  repeated Python/JAX startup dominates small generation phases.
- Persistent workers keep GPU contexts and JIT caches alive across generations,
  which is necessary for 2000-generation experiments.

Smoke / Timing Results:

- Persistent parallel wiring smoke:
  - command used `population.generations=3`, `population.size=2`,
    `population.inner_iters=2`, `population.parallel.enabled=true`,
    `population.parallel.gpu_devices=0,1`, and context-aware crossover;
  - passed;
  - best measured/unweighted loss reached `50742.21461038973`, matching the
    previous non-parallel wiring behavior.
- Persistent parallel 100-step probe:
  - command used `population.generations=2`, `population.size=2`,
    `population.inner_iters=100`, two GPU workers, and context-aware crossover;
  - passed in `real 4.73s`;
  - best measured/unweighted loss was `1859.067646106162`.
- Persistent parallel 2000-generation smoke:
  - command used `population.generations=2000`, `population.size=2`,
    `population.elite_count=1`, `population.inner_iters=100`,
    `population.parallel.enabled=true`, `population.parallel.gpu_devices=0,1`,
    `population.parallel.workers_per_gpu=1`, and context-aware crossover;
  - output directory:
    `outputs/population_parallel_persistent_2000_inner100_smoke`;
  - completed in `real 102.95s`;
  - earlier serial run of the same logical setting took `real 144.71s`;
  - best measured/unweighted loss was identical:
    `1460.0758429369116`;
  - best individual again appeared at generation `73`;
  - output size was `817M`.

Current Performance Judgment:

- GPU-parallel population execution now works and gives a real speedup for the
  long 2000-generation smoke.
- The speedup is not yet close to 2x because every generation still performs
  synchronized phases, parent table reads, crossover construction in the main
  process, metrics/output writes, and many plateaued workers stop after only
  `qdte.stop_patience=15` iterations.
- For larger inner QDTE budgets or larger datasets, the useful fraction of GPU
  work should increase and the parallel speedup should be more meaningful.
- Current 24GB guidance:
  - use `workers_per_gpu=1` by default;
  - try more than one worker per GPU only when `runtime.xla_preallocate=false`
    and measured per-worker VRAM is comfortably below available memory;
  - for Adult/ACS high-throughput configs, start with one worker per GPU and
    `population.size` equal to the number of GPU slots.

Changed Files:

- `scripts/run_population.py`
- `scripts/population_worker.py`
- `qdte/config_validation.py`
- `tests/test_config_validation.py`
- `README.md`
- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m py_compile scripts/run_population.py scripts/population_worker.py qdte/config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest tests/test_config_validation.py -q`
  - `70 passed`
- Persistent parallel 3-generation smoke as described above.
- Persistent parallel 2-generation, 100-step probe as described above.
- Persistent parallel 2000-generation, 100-step smoke as described above.
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q`
  - `180 passed`
- `git diff --check`
  - passed after this handoff update.

Next Recommended Task:

- Add compact output retention for long population runs so final experiments do
  not write hundreds of MB per smoke-scale run.
- Compare persistent parallel `context_aware` crossover against persistent
  parallel `random_row` crossover under equal candidate budget.

## Single-GPU Population Capacity Probe - 2026-06-18

Task:

- Determine how many population workers can run concurrently on one 24GB GPU.
- Clarify that two GPUs with one worker each is not the maximum useful
  population parallelism.

Method:

- Used `population.parallel.enabled=true` with a single visible GPU:
  `CUDA_VISIBLE_DEVICES=0`.
- Set `population.parallel.gpu_devices=0`.
- Set `population.parallel.workers_per_gpu` equal to `population.size`, so all
  population individuals run concurrently on one 24GB card.
- Used `runtime.xla_preallocate=false`.
- Monitored GPU memory with `nvidia-smi`.

Smoke Workload Capacity:

- Config: `configs/smoke.yaml`
- Per run:
  - `population.generations=1`;
  - `population.inner_iters=20`;
  - `population.crossover.enabled=false`;
  - no true-query evaluation or synthetic CSV output.
- Results:
  - `1` worker: peak `979 MiB`, passed;
  - `2` workers: peak `1418 MiB`, passed;
  - `4` workers: peak `2299 MiB`, passed;
  - `8` workers: peak `4058 MiB`, passed;
  - `12` workers: peak `5816 MiB`, passed;
  - `16` workers: peak `7575 MiB`, passed;
  - `24` workers: peak `11094 MiB`, passed;
  - `32` workers: peak `14612 MiB`, passed;
  - `40` workers: peak `18131 MiB`, passed;
  - `48` workers: peak `21650 MiB`, passed;
  - `56` workers: peak `24072 MiB`, failed with JAX
    `RESOURCE_EXHAUSTED`.
- Interpretation:
  - smoke is too small to represent final experiments;
  - for this tiny workload, one 24GB card can run `48` concurrent population
    workers, while `56` is over the memory boundary.

Adult Highpower Capacity:

- Config: `configs/adult_qdte_gpu_highpower.yaml`
- Per run:
  - `population.generations=1`;
  - `population.inner_iters=1`;
  - `population.crossover.enabled=false`;
  - no true-query evaluation or synthetic CSV output.
- Results:
  - `1` worker: peak `6043 MiB`, passed;
  - `2` workers: peak `11066 MiB`, passed;
  - `3` workers: peak `16092 MiB`, passed;
  - `4` workers: peak `21115 MiB`, passed;
  - `5` workers: sampled peak `22042 MiB`, failed with JAX
    `RESOURCE_EXHAUSTED` while allocating an additional `2.97 GiB`.
- Interpretation:
  - for the current Adult high-throughput config, the observed hard limit on a
    24GB 4090 is `4` concurrent population workers per GPU;
  - `5` is not safe because transient JAX allocations exceed remaining memory;
  - recommended stable setting is `3` workers per 24GB GPU for long runs;
  - use `4` only for short probes or when accepting lower headroom.

Current Recommendation:

- For final highpower population experiments on two 24GB GPUs:
  - conservative: `population.size=6`,
    `population.parallel.gpu_devices=0,1`,
    `population.parallel.workers_per_gpu=3`;
  - aggressive: `population.size=8`,
    `population.parallel.gpu_devices=0,1`,
    `population.parallel.workers_per_gpu=4`;
  - do not use `workers_per_gpu=5` for the current Adult highpower config.
- If we want more population workers per GPU, reduce per-worker memory first:
  lower `qdte.total_candidates_per_iter`, lower `qdte.gpu_return_top_k`, reduce
  query block sizes, or use a smaller candidate backend configuration.

Outputs:

- Smoke capacity runs: `outputs/parallel_capacity_smoke`
- Adult highpower capacity runs: `outputs/parallel_capacity_adult_highpower`

Tests / Checks Run:

- Capacity scans above completed.
- No source-code change was made after this probe; only this handoff section
  was added.

## Generational Population Loop And 2000-Generation Smoke - 2026-06-18

Task:

- Add a real multi-generation population loop.
- Run inner QDTE with `population.inner_iters=100`.
- Probe a `2000` generation smoke run on one 24GB GPU.
- Clarify practical population-size limits.

Implementation:

- `scripts/run_population.py` now supports `population.generations`.
- `population.generations=1` preserves the previous one-shot behavior:
  independent restarts, optional crossover children, then final elite ranking.
- `population.generations>1` runs repeated generations:
  - generation `0` initializes `population.size` restart individuals;
  - later generations refill `population.size` slots using previous-generation
    elite clones, optional crossover children, and fresh restarts;
  - every individual reruns inner QDTE for `population.inner_iters` maximum
    iterations using the same measured target through `measurement.reuse_from`;
  - context-aware crossover still shares donor rows only, then rescoring is
    done against the recipient dataset's own residual and edit advantage.
- `qdte/config_validation.py` now validates that
  `population.generations > 0`.

Important Semantics:

- `population.inner_iters=100` is a maximum, not necessarily exactly 100
  accepted/scored iterations. Existing `qdte.stop_patience` still applies; in
  the smoke config it is `15`, so plateaued individuals stop after 15
  no-improvement iterations.
- Current implementation runs population individuals sequentially on the
  selected GPU. Therefore peak VRAM is dominated by one QDTE individual's
  candidate/query buffers, not by `population.size` itself. Larger populations
  mostly cost wall time and disk output unless we later add true concurrent
  population evaluation.

Smoke Runs:

- Multi-generation wiring smoke:
  - command used `population.generations=3`, `population.size=2`,
    `population.inner_iters=2`, and context-aware crossover;
  - passed;
  - best measured/unweighted loss improved from generation `0`
    `103064.22812769201` to generation `2` `50742.21461038973`.
- Single-GPU 100-step probe:
  - command used `CUDA_VISIBLE_DEVICES=0`, `population.generations=2`,
    `population.size=2`, `population.inner_iters=100`, context-aware crossover;
  - passed in `real 3.31s`;
  - best measured/unweighted loss after two generations was
    `1859.067646106162`;
  - observed GPU0 memory peak was `636 MiB` on the smoke workload.
- Full requested smoke:
  - command used `CUDA_VISIBLE_DEVICES=0`, `population.generations=2000`,
    `population.size=2`, `population.elite_count=1`,
    `population.inner_iters=100`, `population.crossover.mode=context_aware`,
    `population.crossover.children=1`,
    `population.crossover.candidates=512`,
    `population.crossover.max_edits=64`,
    `evaluation.compute_true_query_error=false`,
    and `evaluation.save_synthetic_csv=false`;
  - output directory:
    `outputs/population_generations_2000_inner100_smoke`;
  - completed in `real 144.71s`;
  - wrote `2000` generation directories and used `797M` disk;
  - observed GPU0 memory peak was `636 MiB`;
  - best individual was a context-aware crossover child from generation `73`;
  - best measured/unweighted loss was `1460.0758429369116`;
  - generation checkpoints:
    - generation `0`: best `3918.973403222428`;
    - generation `1`: best `1859.067646106162`;
    - generation `2`: best `1573.1067181429053`;
    - generation `5`: best `1501.0721922716284`;
    - generation `10`: best `1466.7009421190405`;
    - generation `50`: best `1460.7796772321844`;
    - generation `73`: best/global best `1460.0758429369116`;
    - generation `1999`: still `1460.0758429369116`.

Judgment:

- For this smoke workload, outer generations help early but plateau quickly.
  After roughly generation `73`, additional generations did not improve the
  measured objective.
- This is consistent with the current inner edit-advantage search reaching a
  local plateau where candidate generation finds no positive-advantage edits.
- The next useful population experiment is not simply more generations; it is
  comparing random-row crossover, context-aware crossover, larger crossover
  candidate pools, and possibly a more exploratory fallback when the positive
  edit rate reaches zero.

24GB Population-Size Guidance:

- With the current sequential wrapper, one 24GB GPU can use large
  `population.size` values from a memory perspective, because individuals are
  evaluated one after another.
- On the smoke workload, peak observed GPU memory was only `636 MiB`; the
  practical limit was wall time and disk.
- A reasonable current single-GPU experimental range is:
  - smoke: `population.size=2` to `16`;
  - larger Adult/ACS-style high-throughput configs: start with
    `population.size=2` to `4`, then increase only after measuring per-run
    VRAM and wall time;
  - avoid very large population sizes until output retention is compacted,
    because `2000` generations with `population.size=2` already wrote `797M`.
- If we later implement concurrent population evaluation, population size will
  need a stricter VRAM formula based on candidate count, query count, sparse
  delta buffers, and table size.

Changed Files:

- `scripts/run_population.py`
- `qdte/config_validation.py`
- `tests/test_config_validation.py`
- `README.md`
- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest tests/test_config_validation.py -q`
  - `69 passed`
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m py_compile scripts/run_population.py qdte/config_validation.py`
  - passed
- 3-generation population smoke as described above.
- 2-generation, 100-step, single-GPU probe as described above.
- 2000-generation, 100-step, single-GPU smoke as described above.
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q`
  - `179 passed`
- `git diff --check`
  - passed after this handoff update.

Next Recommended Task:

- Run the same `population.generations` experiment with `random_row` crossover
  and compare against context-aware crossover under equal candidate budget.
- Add optional compact output retention for long generation runs, because
  storing every generation's full artifacts is unnecessary for final sweeps.

## Context-Aware Population Crossover Direction - 2026-06-18

Question:

- Clarify the current population-level flow.
- Discuss whether random row crossover undermines QDTE's edit-advantage logic.
- Consider a context-aware crossover design.

Current Implementation Clarification:

- Historical note: this section described the wrapper before
  `population.generations` was implemented. The current implementation is
  summarized in the section above.
- At that time, the flow was:
  - measure/project once;
  - run `P` independent QDTE restart individuals for `population.inner_iters`;
  - rank these parents by measured objective;
  - optionally generate row-level random crossover children from ranked parents;
  - run each child for `population.crossover.inner_iters` from the crossed
    table via `init.encoded_npy`;
  - rank parents and children together and output elites.
- Repeated generations where elites seed the next generation through
  clone/crossover/restart have since been implemented through
  `population.generations>1`.

Judgment:

- Random row crossover is useful as a minimal population-search wiring smoke,
  but it is not the most natural final QDTE crossover.
- The raw crossover operation itself is not edit-advantage gated. It may worsen
  the measured objective before the child receives inner QDTE repair.
- Elitism prevents the best measured parent from being lost if parents and
  children are ranked together, so it cannot worsen the selected measured
  objective when extra compute is allowed. This does not imply better true
  offline error or better equal-compute performance.
- For the paper's algorithmic story, a context-aware crossover is more aligned
  with QDTE.

Preferred Context-Aware Crossover:

- Treat crossover as cross-parent row replacement candidates:
  - parent A supplies the current table and old rows;
  - parent B supplies donor rows;
  - a crossover edit is `A[i] -> B[j]`;
  - compute `delta = phi(B[j]) - phi(A[i])`;
  - score it with the same QDTE edit advantage:
    `delta @ residual - 0.5 * ||delta||^2 - lambda_cost * cost`
    under the configured objective weights;
  - accept only positive-advantage non-conflicting row replacements.
- This makes crossover context-aware because it only imports rows from another
  elite dataset when that import helps the current parent's measured residual.
- It preserves the QDTE insight: residuals define a direction field and edits
  are accepted by exact measured-objective improvement.

Next Recommended Implementation:

- Implemented an optional `context_aware` crossover mode in
  `scripts/run_population.py`:
  - generate random donor row pairs from elite parent pairs;
  - score them with existing QDTE candidate scoring;
  - apply top non-conflicting positive replacements to form a child;
  - run short inner QDTE repair from that child;
  - compare against the current random row crossover.
- Candidate pool semantics:
  - each dataset/individual maintains its own candidate scoring context,
    residual, answer vector, and measured objective;
  - context-aware crossover shares donor rows across parent datasets, not
    already-scored advantages;
  - each donor replacement is rescored against the recipient dataset's
    residual.
- Interpretation:
  - random row crossover remains a useful search-space expansion ablation;
  - context-aware crossover is the QDTE-aligned version because crossover edits
    are accepted only through the recipient dataset's edit advantage;
  - it may still fail to beat random crossover under equal compute because it
    filters through the current local residual direction, while random
    crossover can inject larger disruptive moves for the inner QDTE loop to
    repair.

Changed Files:

- `docs/HANDOFF.md`
- `scripts/run_population.py`
- `qdte/config_validation.py`
- `tests/test_config_validation.py`
- `README.md`
- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`

Tests Run:

- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest tests/test_config_validation.py -q`
  - `68 passed`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/envs/qdte/bin/python scripts/run_population.py --config configs/smoke.yaml --run.output_dir outputs/population_context_crossover_smoke --population.size 2 --population.elite_count 1 --population.inner_iters 2 --population.evaluate_individuals false --population.crossover.enabled true --population.crossover.mode context_aware --population.crossover.children 1 --population.crossover.fraction 0.5 --population.crossover.candidates 512 --population.crossover.max_edits 64 --population.crossover.inner_iters 2 --evaluation.compute_true_query_error false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false --runtime.log_measurement_groups false --qdte.log_every 1 --qdte.candidate_diagnostics false`
  - passed;
  - parents had final measured losses `103064.22812769201` and
    `104595.27962576662`;
  - context-aware child scored `511` donor-row candidates, found `277`
    positive candidates, selected and accepted `64` edits, with batch advantage
    `23863.69140625`;
  - child initial measured loss was `79200.5`;
  - after two inner QDTE iterations, child final measured/unweighted loss was
    `71737.88779669191`, better than both parents in this wiring smoke.
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m py_compile scripts/run_population.py qdte/evolution/engine.py qdte/config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q`
  - `178 passed`
- `git diff --check`
  - passed after this handoff update.

Next Recommended Task:

- Compare single-layer QDTE, random-restart elite selection, random-row
  crossover, and context-aware crossover under an equal total candidate budget.
- If population search remains useful, implement repeated generations where
  elites seed the next round instead of the current one-shot parents plus
  crossover children wrapper.

## Population-Level Evolution Design Discussion - 2026-06-18

Task:

- Discuss how to add an outer population-level evolutionary layer on top of the
  current single-dataset QDTE generator.

Design Direction:

- Treat population evolution as DP post-processing over one fixed measured
  artifact:
  - schema;
  - query catalogue;
  - workload groups;
  - projected target;
  - objective weights, currently `objective_weighting=unweighted`;
  - measurement metadata.
- Do not remeasure real data inside the population loop.
- Do not use offline true answers for selection, crossover, restart, stopping,
  or hyperparameter choice.

Recommended Two-Level Structure:

- Inner layer:
  - current single-dataset QDTE;
  - residual-directed row edits;
  - constructive pair / transport;
  - exact measured-objective edit advantage.
- Outer layer:
  - maintain a population of synthetic datasets, each with its own `X_syn`,
    `answer_syn`, `residual`, loss, RNG stream, and age/stall counters;
  - run short inner QDTE phases for each individual;
  - rank individuals by measured objective only;
  - keep elites;
  - refill the population with elite clones, independent restarts, and
    row/block crossover between elite parents.

Preferred Initial Implementation:

- Start with a simple island/elite wrapper and simple row-level crossover
  before implementing more complicated recombination:
  - population size `P`;
  - elite count `E`;
  - inner phase budget in candidate evaluations, not only iterations;
  - clone top elites;
  - random-restart the worst individuals;
  - optionally crossover two strong parents by swapping random row subsets,
    then continue inner QDTE from the child table.
- Use equal total candidate-evaluation budget when comparing against inner-only
  QDTE, because population search otherwise wins by spending more compute.

Crossover Notes:

- Row-level crossover is the safest first crossover:
  - choose two parent synthetic tables with the same row count and schema;
  - choose a public/random subset of row indices or blocks;
  - child rows come from parent A outside the mask and parent B inside the mask;
  - recompute `answer_syn`, residual, and measured loss for the child.
- It is simple and valid DP post-processing.
- It may not always improve because row mixtures can break marginal structure;
  therefore clone+restart should be the first baseline, with crossover as an
  ablation.

Key Comparison Variants:

- Inner-only QDTE under the same total candidate budget.
- Random-restart ensemble: run several independent single-layer QDTE instances
  and keep the best measured objective at the end.
- Population QDTE without crossover: elite selection plus clone/restart.
- Population QDTE with row/block crossover.
- Population random mutation / upstream Private-GSD baseline.

Current Recommendation:

- Initial implementation status:
  - measured-target reuse boundary is implemented through
    `measurement.reuse_from`;
  - `init.encoded_npy` lets `run_qdte` start from an externally supplied encoded
    synthetic table;
  - `scripts/run_population.py` implements a first population-level
    random-restart/elite wrapper over one shared measurement artifact;
  - `scripts/run_population.py` now supports minimal row-level random
    crossover children via `population.crossover.enabled=true`;
  - inter-generation clone/restart cycles and more structured block crossover
    are not implemented yet.
- Continue in two stages:
  1. add inter-generation clone/restart phases if the random-restart/crossover
     wrapper is useful;
  2. add structured row-block crossover as a separate ablation.
- This keeps the innovation centered on directed inner edits while allowing a
  fair Private-GSD-style population comparison.

Changed Files:

- `docs/HANDOFF.md`
- `qdte/measurement/measure.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_population.py`
- `tests/test_config_validation.py`
- `tests/test_engine_smoke.py`
- `README.md`
- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`

Tests Run:

- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - `74 passed`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/envs/qdte/bin/python scripts/run_population.py --config configs/smoke.yaml --run.output_dir outputs/population_smoke --population.size 2 --population.elite_count 1 --population.inner_iters 2 --population.evaluate_individuals false --evaluation.compute_true_query_error false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false --runtime.log_measurement_groups false --qdte.log_every 1 --qdte.candidate_diagnostics false`
  - passed;
  - reused `outputs/population_smoke/measurement/measurements.json`;
  - wrote `outputs/population_smoke/population_summary.json`;
  - best individual was `individual_000`, final measured/unweighted loss
    `103064.22812769201`.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/envs/qdte/bin/python scripts/run_population.py --config configs/smoke.yaml --run.output_dir outputs/population_crossover_smoke --population.size 2 --population.elite_count 1 --population.inner_iters 2 --population.evaluate_individuals false --population.crossover.enabled true --population.crossover.children 1 --population.crossover.fraction 0.5 --population.crossover.inner_iters 2 --evaluation.compute_true_query_error false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false --runtime.log_measurement_groups false --qdte.log_every 1 --qdte.candidate_diagnostics false`
  - passed;
  - generated one row-level random crossover child;
  - child used parents `(0, 1)`, swapped `512` rows, and continued for `2`
    inner QDTE iterations from `init.encoded_npy`;
  - best individual was the crossover child, final measured/unweighted loss
    `91603.75887876698`.
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q`
  - `178 passed`
- `git diff --check`
  - passed after this handoff update.

Next Recommended Task:

- Add inter-generation clone/restart if we want population evolution beyond
  one-shot random-restart plus crossover child generation, then add structured
  row-block crossover as a separate ablation.

## Single-Layer Closeout Defaults And Private-GSD Baseline Clarification - 2026-06-18

Task:

- Close out immediate single-layer QDTE cleanup before formal multi-seed and
  large-dataset experiments.
- Clarify what "exact Private-GSD baseline" means, given that upstream
  open-source Private-GSD code is available.

Changes:

- Set main experiment configs to use `qdte.objective_weighting: unweighted`:
  - `configs/smoke.yaml`;
  - `configs/adult_qdte.yaml`;
  - `configs/adult_qdte_gpu_highpower.yaml`;
  - `configs/acs_qdte.yaml`.
- Updated README and architecture docs to state that unweighted objective is
  the current single-layer QDTE mainline.
- Updated final experiment/ablation docs to clarify the Private-GSD baseline
  wording.

Private-GSD Baseline Clarification:

- Yes, the preferred paper-level Private-GSD baseline should use the upstream
  open-source implementation directly, via a thin adapter.
- "Exact baseline" does not mean rewriting Private-GSD inside QDTE.
- It means the comparison should preserve the upstream algorithm's intended
  population-level mutation/selection procedure and align public inputs as much
  as possible:
  - preprocessing/binning;
  - workload/statistic definitions, or a clearly documented closest supported
    statistic interface;
  - privacy budget/noise setting;
  - synthetic row count;
  - seeds;
  - compute budget/wall time/candidate evaluations where applicable.
- Existing internal variants such as `pgsd_style_mutate50` are useful sanity
  baselines, but they are not an upstream Private-GSD result because they use
  QDTE's measurement/workload stack and a simplified mutate-only procedure.

Current Status:

- Single-layer QDTE is now configured around the simpler post-projection
  objective: directly fit the feasible projected target with unweighted raw
  residuals.
- Multi-seed and large-dataset runs are intentionally deferred to formal
  experiments.
- Outer population evolution remains unimplemented and optional.

Changed Files:

- `configs/smoke.yaml`
- `configs/adult_qdte.yaml`
- `configs/adult_qdte_gpu_highpower.yaml`
- `configs/acs_qdte.yaml`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/envs/qdte/bin/python - <<'PY' ...`
  - loaded and validated all YAML configs under `configs/`;
  - confirmed all four main configs set `objective_weighting=unweighted`.
- `git diff --check`
  - passed.

Next Recommended Task:

- Before formal runs, add or keep a thin adapter script that invokes upstream
  Private-GSD from `/home/qianqiu/my_life/baseline/private-de/private-gsd` and
  writes results into the same summary table format as QDTE.

## Single-Layer Versus Outer Population Evolution - 2026-06-18

Question:

- Whether the two-level evolution framework has already been implemented:
  GPU-parallel evolution of several synthetic datasets, dataset-level
  crossover/recombination, measured-loss driven elite selection, and continued
  inner QDTE evolution.
- Whether the single-dataset evolution layer is now conceptually complete.

Current Code Status:

- The outer population layer is not yet implemented as a first-class algorithm.
- Current code has:
  - one `QDTEState` per `run_qdte` invocation;
  - one synthetic table evolved by residual-directed candidate generation,
    edit advantage, and transport;
  - robustness scripts that launch multiple independent runs/variants;
  - internal Private-GSD-style random mutation ablations.
- Current code does not yet have:
  - a population of multiple `QDTEState` objects inside one run;
  - GPU-batched evolution of several synthetic tables as parallel individuals;
  - row/block crossover between synthetic datasets;
  - elite selection and refill/restart inside a shared measured-target run;
  - a reusable measured-target artifact consumed by many population
    individuals.

Single-Layer Judgment:

- The single-dataset QDTE generation story is now mostly complete:
  - feasible consistency-projected DP target;
  - orthogonal workload construction to reduce measurement sensitivity;
  - residual-directed local edit construction;
  - exact local measured-loss edit advantage;
  - constructive pair / transport-style multi-edit acceptance;
  - GPU fused candidate generation/scoring, including halfspace support;
  - unweighted objective as the current main post-projection generation
    objective;
  - measured, unweighted, per-family, heldout, and offline true-error metrics.
- Remaining single-layer work is mostly experimental validation and packaging,
  not a missing core mechanism:
  - multi-seed/full-scale confirmation;
  - main config/experiment defaults aligned to `objective_weighting=unweighted`;
  - final ablation matrix cleanup;
  - exact upstream Private-GSD comparison if needed.

Outer-Layer Interpretation:

- Adding an outer population loop would make the method a genuine two-level
  evolutionary framework.
- The likely narrative would be:
  - inner layer: individual/table-row-level directed evolution from residuals;
  - outer layer: dataset-population selection/crossover/restart over synthetic
    tables using the same measured objective.
- This should be treated as an extension or next build, not something already
  completed.
- It should be implemented as DP post-processing over one fixed measured target:
  no remeasurement of real data, and no offline true metrics for elite
  selection, crossover, restart, stopping, or hyperparameter choice.

Minimal Implementation Plan:

- First expose or wrap a generator entry point that consumes an already measured
  target artifact:
  - schema;
  - query catalogue;
  - workload groups;
  - projected target;
  - objective weights;
  - measurement metadata.
- Then add an outer population driver:
  - initialize `P` synthetic tables;
  - run short inner QDTE phases per individual;
  - score by measured/unweighted objective only;
  - keep elites;
  - refill with elite clones, random restarts, and row/block crossover;
  - compare against inner-only QDTE under equal total candidate evaluations and
    equal wall time.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation-only status update:
  - `git diff --check`

Next Recommended Task:

- If we want the two-level algorithm in the paper, implement the measured-target
  reuse boundary first, then the outer population driver.
- If we want to keep the contribution focused, keep the outer population loop as
  an optional extension and make the single-dataset directed evolution the main
  algorithm.

## Unweighted Objective Ablation - 2026-06-18

Task:

- Add and test an optimization mode that does not use inverse-variance weights
  after consistency projection.
- Answer whether QDTE can optimize raw residuals against the feasible projected
  target instead of optimizing standardized residuals.

Implementation:

- Added `qdte.objective_weighting`:
  - `variance` keeps the previous behavior and uses
    `measurements.inv_variances`;
  - `unweighted` sets optimization-time `variance=1`, `inv_variance=1`,
    and `sigma=1` for measured loss, edit advantage, active-query selection,
    candidate scoring, transport, and debt updates.
- Measurement variances are still stored and reported separately as
  `measurement_*` metrics.
- Final metrics/runtime now report:
  - `objective_weighting`;
  - `objective_variance_*`;
  - `objective_inv_variance_*`;
  - `measurement_variance_*`;
  - `measurement_inv_variance_*`.
- This preserves the QDTE objective invariant because the configured objective
  still computes
  `measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]`; in
  unweighted mode `inv_variance[q]=1`.
- DP boundary is unchanged: in `privacy.mode=dp`, optimization still uses only
  noisy/projected measurements. Exact true answers are only used for offline
  evaluation metrics.

Clarification:

- Consistency projection makes the measured answers feasible/consistent, but it
  does not generally prove that every query has identical post-projection
  variance.
- Projection can create query-dependent effective variance and cross-query
  covariance, especially with clipping, active feasible constraints, and
  overlapping query scopes.
- Still, if the algorithmic goal is to match the feasible projected target as a
  deterministic object, unweighted optimization is a meaningful ablation and
  may be the cleaner objective for the post-projection generation stage.

Smoke Experiments:

- Both runs used `constructive_pair`, `max_iters=2000`,
  `stop_patience=2000`, `local_table_feasible_jax`, and
  `qdte.objective_weighting=unweighted`.
- `projection.uncertainty.enabled=false` because variance estimates are ignored
  by this objective and the projected target is unchanged.

| Workload | Run | Queries | Final objective loss | Final unweighted loss | Final true RMSE | Accepted edits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original mixed | Raw variance weighted | `243` | `2.318852357` | `75.859848591` | `0.002727697` | `1458` |
| Original mixed | Projection-aware variance weighted | `243` | `8.176284459` | `124.201921078` | `0.002725433` | `1493` |
| Original mixed | Unweighted objective | `243` | `56.984528038` | `56.984528038` | `0.002698117` | `1365` |
| Orthogonal mixed | Raw variance weighted | `242` | `3.976314087` | `47.040555620` | `0.002338149` | `942` |
| Orthogonal mixed | Projection-aware variance weighted | `242` | `4.215038075` | `21.917599106` | `0.002305222` | `1033` |
| Orthogonal mixed | Unweighted objective | `242` | `21.849774742` | `21.849774742` | `0.002279082` | `955` |

Family-Level Smoke Results:

| Workload family | Run | Family queries | Final family unweighted loss | Family true RMSE |
| --- | --- | ---: | ---: | ---: |
| `mixed` | Raw variance weighted | `45` | `62.692866258` | `0.004242641` |
| `mixed` | Projection-aware variance weighted | `45` | `103.767566613` | `0.004297286` |
| `mixed` | Unweighted objective | `45` | `25.594648771` | `0.004123106` |
| `orthogonal_kway_mixed` | Raw variance weighted | `44` | `15.726940881` | `0.002969542` |
| `orthogonal_kway_mixed` | Projection-aware variance weighted | `44` | `7.194050800` | `0.002705214` |
| `orthogonal_kway_mixed` | Unweighted objective | `44` | `5.096493732` | `0.002730301` |

Current Judgment:

- The unweighted objective is worth keeping.
- In these smoke runs, it gives the best total true RMSE on both original mixed
  and orthogonal mixed.
- For the orthogonal family itself, it improves raw residual fit but is slightly
  worse than projection-aware variance weighting by family true RMSE; this
  needs multi-seed confirmation.
- This suggests objective weighting is a real hyperparameter/design axis, not
  just a reporting choice.
- Current mainline decision:
  - use `qdte.objective_weighting=unweighted` as the default experimental
    narrative for the generation stage after feasible consistency projection;
  - keep inverse-variance and projection-aware variance weighting as ablations,
    not as the main algorithm path, unless multi-seed/full-scale experiments
    overturn this smoke-run pattern.
- Rationale:
  - it directly optimizes the feasible projected target;
  - it is simpler to explain than choosing a diagonal variance approximation
    after a biased feasible projection;
  - current smoke evidence is better on both raw residual loss and total true
    RMSE.

Changed Files:

- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `tests/test_engine_smoke.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests Run:

- Documentation-only judgment update:
  - `git diff --check`
  - no code tests rerun for this note.
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - `69 passed`
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q`
  - `173 passed`
- `git diff --check`
  - passed after this documentation update.

Experiment Outputs:

- `outputs/exp_unweightedobj_original_consistency2000_constructive_pair/`
- `outputs/exp_unweightedobj_orthogonal_consistency2000_constructive_pair/`

Next Recommended Task:

- Run a multi-seed objective-weighting sweep:
  - `variance` with raw Gaussian variance;
  - `variance` with projection-aware diagonal variance;
  - `unweighted`;
  - optionally tempered weights `inv_variance^alpha` for
    `alpha in {0.25, 0.5, 0.75}`.
- Report total true RMSE, family true RMSE, unweighted measured loss, weighted
  measured loss, and accepted edit counts together.

## Raw Variance Versus Projection-Aware Variance - 2026-06-18

Task:

- Clarify how `variance-only` differs from the earlier weighted loss that
  directly divided by Gaussian variance.

Clarification:

- Both methods use the same weighted-loss form:
  `measured_loss = 0.5 * sum_q residual[q]^2 / variance[q]`.
- The difference is which variance is used.
- Earlier/raw variance:
  - `variance[q] = sigma_group(q)^2` from the Gaussian mechanism;
  - this is the variance before clipping/consistency/feasible projection;
  - it ignores that projection changes marginal uncertainty and correlates
    query errors.
- Projection-aware variance-only:
  - target remains the same feasible projected target `P(y)`;
  - no bias is subtracted;
  - bootstrap estimates a diagonal approximation to
    `Var(P(center + Gaussian noise))[q]`;
  - `variance[q]` is replaced by this effective post-projection variance.

Observed Smoke Results:

| Workload | Weighting | Final weighted loss | Final unweighted loss | Final true RMSE |
| --- | --- | ---: | ---: | ---: |
| Original mixed | Raw Gaussian variance | `2.318852357` | `75.859848591` | `0.002727697` |
| Original mixed | Projection-aware variance-only | `8.176284459` | `124.201921078` | `0.002725433` |
| Orthogonal mixed | Raw Gaussian variance | `3.976314087` | `47.040555620` | `0.002338149` |
| Orthogonal mixed | Projection-aware variance-only | `4.215038075` | `21.917599106` | `0.002305222` |

Interpretation:

- Weighted loss is not directly comparable across raw variance and
  projection-aware variance-only, because the denominator changes.
- On original random mixed:
  - true RMSE is essentially unchanged;
  - unweighted measured loss is worse under projection-aware variance.
- On orthogonal mixed:
  - projection-aware variance-only is better by both unweighted measured loss
    and true RMSE.
- Current empirical judgment:
  - for the orthogonal workload, projection-aware variance-only is better than
    raw Gaussian variance;
  - for original mixed, it is not clearly better;
  - therefore the strongest current setting remains orthogonal workload plus
    projection-aware variance-only.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation-only update; run `git diff --check` before closing.

Next Recommended Task:

- In multi-seed experiments, report raw Gaussian variance and
  projection-aware variance-only side by side, using true RMSE and unweighted
  residual metrics as the fair comparison axes.

## Debias-Reproject Judgment Clarification - 2026-06-18

Task:

- Clarify whether debias+reproject is a new bias estimator or just a feasibility
  repair, and how it compares with the Gaussian/oracle bias diagnostic.

Clarification:

- `debias + reproject` is not a new bias estimator.
- The bias estimate is still the same plug-in bootstrap estimate:
  `b_hat = mean_i(P(P(y) + noise_i)) - P(y)`.
- Reprojection only does:
  `target = P(P(y) - alpha * b_hat)`.
- Therefore it should not be presented as a principled exact debias method.
  It is a diagnostic/ablation that answers:
  "If direct debias fails because it leaves the feasible region, does putting
  it back into the feasible region help?"

Current Evidence:

- Direct plug-in debias is worse than variance-only:
  - final true RMSE: `0.002326633` vs `0.002305222`;
  - final target error L2: `35.871482769` vs `35.026265157`;
  - negative targets: 6.
- Full debias + reproject fixes infeasibility and lowers measured loss:
  - negative targets: 0;
  - weighted loss: `3.800782531`;
  - but final target error L2 is still worse than variance-only:
    `35.452127117` vs `35.026265157`;
  - true RMSE is effectively tied/slightly worse than variance-only:
    `0.002307014` vs `0.002305222`.
- Alpha 0.5 debias + reproject is the only promising row in this smoke run:
  - final target error L2: `34.988361178`, slightly better than
    variance-only;
  - final true RMSE: `0.002261792`, best in this four-way comparison.
- The plug-in bias itself is not accurate against the oracle MC diagnostic:
  - plug-in/oracle cosine: `-0.028660775`;
  - plug-in minus oracle bias L2: `13.960717541`.

Judgment:

- Full plug-in debias + reproject is conceptually clumsy and should not be the
  main method.
- It is useful as a negative/control ablation because it shows feasibility
  matters.
- The only potentially useful route is partial feasible debias:
  shrink the questionable plug-in correction (`alpha < 1`) and reproject to
  keep the target realizable.
- The mainline should remain variance-only unless an alpha sweep across seeds
  shows partial feasible debias is consistently better.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation-only update; run `git diff --check` before closing.

Next Recommended Task:

- Run multi-seed alpha sweep before treating partial debias as anything more
  than a promising ablation.

## Debias Variant Explanation And Current Interpretation - 2026-06-18

Task:

- Explain how the current debias variants are computed.
- Clarify why reprojecting after debias is not a route to unbiasedness.
- Clarify why biased feasible projection can still outperform an unbiased
  equality-only consistency projection.

Debias Variants:

- Variance-only:
  - noisy measurement: `y = true_answers + noise`;
  - feasible projection: `p = P(y)`;
  - optimized target: `target = p`;
  - bootstrap only estimates post-projection diagonal variances for weights.
- Plug-in debias:
  - center: `c = p` when `projection.uncertainty.center=projected`;
  - bootstrap samples: `z_i = c + noise_i`;
  - projected samples: `p_i = P(z_i)`;
  - plug-in bias: `b_hat = mean_i(p_i) - c`;
  - optimized target: `target = p - alpha * b_hat`;
  - with `alpha=1`, this is full plug-in debias.
- Debias then reproject:
  - optimized target before feasibility repair:
    `target_raw = p - alpha * b_hat`;
  - final target: `target = P(target_raw)`;
  - this is a biased feasible variant, not an unbiased estimator.
- Oracle-bias diagnostic:
  - offline only, using true answers `t`;
  - samples: `z_i = t + noise_i`;
  - oracle MC bias: `b_oracle = mean_i(P(z_i)) - t`;
  - used only to compare plug-in bias with true Monte Carlo bias in analysis.

Why Reproject:

- Direct plug-in debias can produce negative query targets or violate local
  table feasibility.
- Synthetic-data query answers cannot realize negative counts or infeasible
  cross-scope marginals.
- Reprojection maps the debiased vector back into the feasible answer set that
  QDTE can approximately fit.
- Reprojection reintroduces bias. It should be understood as a lower-MSE
  feasible post-processing heuristic, not as unbiased correction.

Why Biased Feasible Projection Can Beat Unbiased Equality Projection:

- Unbiasedness alone controls the mean of an estimator, not its finite-sample
  squared error or optimization usefulness.
- Feasible projection can lower MSE by reducing variance enough to offset
  added bias:
  `MSE = variance + bias^2`.
- Equality-only unbiased projection can preserve an unbiased target while still
  leaving the query vector outside the feasible synthetic-data answer set or
  with inconsistent local edit directions.
- QDTE optimizes a finite synthetic dataset. If the target is unbiased but not
  feasible/easy to realize, the residual field can be harder to follow and the
  final measured loss/true RMSE can be worse.
- Nonnegative/local-table projection is biased, but it gives a feasible target
  and a cleaner optimization geometry. This is consistent with the expert
  claim that biased feasible projection can have lower overall query error.

Current Interpretation:

- The current plug-in bias estimate is not reliable as exact debias:
  `plugin/oracle bias cosine ~= -0.029` on the orthogonal smoke run.
- Projection-aware variance-only can perform well because it keeps the feasible
  target `P(y)` and only changes weighting by post-projection uncertainty.
- Partial debias plus reproject is promising because it applies a conservative
  bias correction while preserving feasibility.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation-only update; run `git diff --check` before closing the task.

Current Status:

- Debias variants and the biased-feasible-versus-unbiased-equality distinction
  are documented.

Next Recommended Task:

- Run the planned `debias_alpha` sweep with `reproject_debiased_target=true`
  over multiple seeds before deciding whether partial feasible debias should
  enter the main method.

## Oracle Bias Diagnostic And Debias-Reproject Ablation - 2026-06-18

Task:

- Add offline true/oracle projection-bias diagnostics for analysis only.
- Add debias controls:
  - `projection.uncertainty.debias_alpha`;
  - `projection.uncertainty.reproject_debiased_target`.
- Run orthogonal 2000-step smoke ablations comparing variance-only,
  plug-in debias, full debias+reproject, and partial debias+reproject.

Implementation:

- `qdte/measurement/measure.py`
  - Keeps the plug-in bootstrap bias vector in memory as
    `Measurements.projection_uncertainty_bias`.
  - Does not write this vector to `measurements.json`.
  - Adds `projection.uncertainty.debias_alpha`, default `1.0`.
  - Adds `projection.uncertainty.reproject_debiased_target`, default `false`.
  - If `debias_target=true`, the target update is now:
    `target = projected - debias_alpha * plugin_bias`.
  - If `reproject_debiased_target=true`, the debiased target is passed through
    the same configured feasible projection again.
- `qdte/evolution/engine.py`
  - Adds evaluation-only `evaluation.oracle_projection_bias`.
  - When enabled and exact true answers are already being computed for offline
    evaluation, it estimates:
    `oracle_bias ~= mean_i P(true_answers + Gaussian_noise_i) - true_answers`.
  - Writes aggregate diagnostics to `oracle_projection_bias.json` and scalar
    summaries into `metrics_final.json`.
  - This does not feed into measurement, target construction, candidate
    generation, scoring, stopping, transport, or hyperparameter selection.
- `qdte/config_validation.py`
  - Validates `debias_alpha in [0, 1]`.
  - Validates `evaluation.oracle_projection_bias.num_samples > 1`.

New Config Keys:

- `projection.uncertainty.debias_alpha`: partial plug-in debias strength.
- `projection.uncertainty.reproject_debiased_target`: enforce feasibility after
  debias.
- `evaluation.oracle_projection_bias.enabled`: enable offline oracle-bias
  diagnostics.
- `evaluation.oracle_projection_bias.num_samples`: Monte Carlo samples.
- `evaluation.oracle_projection_bias.seed`: optional diagnostic seed.

Smoke Setup:

- Base config: `configs/smoke.yaml`.
- Workload: orthogonal mixed enabled, original mixed disabled.
- Projection: `local_table_feasible_jax`.
- Projection uncertainty:
  - `bootstrap_diagonal`;
  - `num_samples=16`;
  - `center=projected`;
  - `min_raw_variance_fraction=0.05`.
- QDTE:
  - `variant=constructive_pair`;
  - `max_iters=2000`;
  - `stop_patience=2000`.
- Oracle diagnostic:
  - `evaluation.oracle_projection_bias.enabled=true`;
  - `num_samples=16`;
  - `seed=4242`.

Output Directories:

- `outputs/exp_biasdiag_orthogonal_varonly16_consistency2000_constructive_pair`
- `outputs/exp_biasdiag_orthogonal_plugindebias16_consistency2000_constructive_pair`
- `outputs/exp_biasdiag_orthogonal_debias_reproject16_consistency2000_constructive_pair`
- `outputs/exp_biasdiag_orthogonal_debias_alpha05_reproject16_consistency2000_constructive_pair`

Main Results:

| Run | Final weighted loss | Final unweighted loss | Final true RMSE | Orthogonal-family true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Variance-only | `4.215038075` | `21.917599106` | `0.002305222` | `0.002705214` | 1033 |
| Plug-in debias | `6.047716294` | `28.502779504` | `0.002326633` | `0.002540580` | 1014 |
| Full debias + reproject | `3.800782531` | `19.334146917` | `0.002307014` | `0.002486326` | 1032 |
| Alpha 0.5 debias + reproject | `3.920276457` | `21.148945220` | `0.002261792` | `0.002504541` | 1014 |

Target Feasibility:

| Run | Pre-reproject min target | Pre-reproject negatives | Final min target | Final negatives |
| --- | ---: | ---: | ---: | ---: |
| Variance-only | `0.0` | 0 | `0.0` | 0 |
| Plug-in debias | `-1.035898853` | 6 | `-1.035898853` | 6 |
| Full debias + reproject | `-1.035898853` | 6 | `0.0` | 0 |
| Alpha 0.5 debias + reproject | `-0.517949427` | 6 | `0.0` | 0 |

Oracle Bias Diagnostics:

| Metric | Value |
| --- | ---: |
| Oracle MC bias L2 | `10.460274952` |
| Oracle MC bias Linf | `1.991957664` |
| Plug-in bias L2 | `8.950830765` |
| Plug-in bias Linf | `1.489848614` |
| Plug-in minus oracle bias L2 | `13.960717541` |
| Plug-in minus oracle bias Linf | `2.293832310` |
| Plug-in/oracle bias cosine | `-0.028660775` |

Target Error Against True Answers:

| Run | Observed projected target error L2 | Final optimized target error L2 |
| --- | ---: | ---: |
| Variance-only | `35.026265157` | `35.026265157` |
| Plug-in debias | `35.026265157` | `35.871482769` |
| Full debias + reproject | `35.026265157` | `35.452127117` |
| Alpha 0.5 debias + reproject | `35.026265157` | `34.988361178` |

Interpretation:

- Direct plug-in debias is not a reliable exact bias correction:
  - it creates negative optimized targets;
  - it increases final target error L2;
  - its bias vector is almost orthogonal to the oracle MC bias
    (`cosine ~= -0.029`);
  - the plug-in/oracle difference L2 is larger than either bias vector alone.
- Reprojecting after debias fixes feasibility:
  - negative targets go from 6 to 0;
  - measured loss improves strongly.
- Full debias + reproject gets the best measured loss, but not the best true
  RMSE.
- Alpha 0.5 debias + reproject is the best true-RMSE row in this smoke run:
  `0.002261792`.
- This suggests partial feasible debias may be a useful ablation, but the
  current plug-in bias should not be presented as a mathematically exact
  correction.

Current Judgment:

- Mainline remains projection-aware variance plus orthogonal measurement.
- Debias should be reported as a controlled ablation:
  - direct plug-in debias is unsafe;
  - full debias + reproject is feasible and lowers measured loss;
  - partial debias + reproject is currently the most promising debias variant
    by true RMSE on this seed.
- The oracle bias diagnostic is valuable for analysis, but must remain
  evaluation-only.

Changed Files:

- `qdte/measurement/measure.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `tests/test_measurement.py`
- `tests/test_engine_smoke.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m py_compile qdte/measurement/measure.py qdte/config_validation.py qdte/evolution/engine.py tests/test_measurement.py tests/test_config_validation.py tests/test_engine_smoke.py`
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q tests/test_measurement.py tests/test_config_validation.py tests/test_engine_smoke.py::test_engine_smoke_outputs`
  - `78 passed in 2.88s`
- `/home/qianqiu/.anaconda3/envs/qdte/bin/python -m pytest -q`
  - `171 passed in 10.58s`
- `git diff --check`
  - passed.
- Four 2000-step orthogonal smoke ablations listed above.

Current Status:

- Oracle bias diagnostics are implemented and tested.
- Debias-alpha and debias-reproject variants are implemented and tested.
- Smoke evidence now supports testing partial feasible debias further, rather
  than using direct plug-in debias.

Next Recommended Task:

- Run a small alpha sweep under orthogonal workload:
  - `debias_alpha in {0.0, 0.25, 0.5, 0.75, 1.0}`;
  - always with `reproject_debiased_target=true`;
  - multiple seeds.
- Keep direct `debias_target=true` without reproject as a negative/control
  ablation.

## Exact Bias Versus Feasible Reprojection - 2026-06-18

Task:

- Clarify whether we should compute exact projection bias, and what
  "debias then reproject" means.

Clarification:

- The exact bias of a projection estimator is:
  `b(t) = E_noise[P(t + noise)] - t`,
  where `P` is the configured weighted projection operator and `t` is the
  unknown true query-answer vector.
- This exact bias depends on the unknown true answer `t`. Therefore it cannot
  be used inside `privacy.mode=dp` optimization, active query selection,
  candidate generation, scoring, stopping, or hyperparameter selection.
- It can be computed or approximated only as an offline diagnostic in smoke
  experiments where exact true answers are available for evaluation.
- The observed clipped negative amount from one noisy realization is not the
  same as statistical bias. Even for scalar clipping at zero,
  `P(y)=max(y,0)`, the bias is a function of `t/sigma`, not just the observed
  negative part of `y`.

Meaning Of "Debias Then Reproject":

- Current plug-in debias does:
  `target = P(noisy) - estimated_bias`.
- This can make the optimized target infeasible, for example producing
  negative query counts.
- "Debias then reproject" would do:
  `target = P(P(noisy) - estimated_bias)`.
- This is not an unbiased estimator. It is a biased feasible post-processing
  variant intended to test whether the current debias failure is mainly caused
  by infeasible targets.
- This does not contradict the earlier point that exact unbiasedness is
  generally impossible when we require nonnegative feasible outputs under
  two-sided Gaussian/discrete-Gaussian noise.

Current Recommendation:

- Add an offline oracle-bias diagnostic:
  - compute true query answers `t` only for evaluation;
  - repeatedly draw fresh Gaussian noise according to the measured variances;
  - apply the exact configured projection operator;
  - estimate `E[P(t + noise)] - t`;
  - compare this oracle bias with the current plug-in bootstrap bias.
- Keep this diagnostic outside the DP optimization path.
- For optimization ablations, test:
  - variance-only;
  - plug-in debias;
  - plug-in debias then reproject;
  - partial debias then reproject.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation-only clarification.

Current Status:

- Exact bias is useful for diagnosis and paper analysis, but not available as a
  valid DP optimization signal.
- Reprojection after debias is a feasible biased variant, not a route to exact
  unbiasedness.

Next Recommended Task:

- Implement oracle-bias diagnostics in the evaluation path only, then compare
  oracle bias with current plug-in bootstrap bias on the smoke workload.

## Debias Interpretation And Concern - 2026-06-18

Task:

- Clarify why `projection-aware variance + debias_target=false` can outperform
  `debias_target=true`, even though the feasible projection is biased by
  nonnegativity/lower-bound constraints.

Current Finding:

- The current debias implementation is a plug-in bootstrap correction, not an
  exact mathematical removal of projection bias.
- It estimates:
  `bias ~= mean(project(center + Gaussian noise)) - center`,
  where `center` is usually the already projected target.
- If `debias_target=true`, the optimized target is changed to:
  `target_projected := projected - bias`.
- This is useful as an ablation, but it can move the target outside the
  feasible answer region that synthetic data can realize.

Observed Diagnostics:

| Run | Min target after debias/variance | Negative targets | Bias L2 | Bias Linf |
| --- | ---: | ---: | ---: | ---: |
| Var-only original mixed | `0.0` | 0 | `14.795448557` | `5.459616661` |
| Debias original mixed | `-3.420611858` | 9 | `14.795448557` | `5.459616661` |
| Var-only orthogonal mixed | `0.0` | 0 | `8.950830753` | `1.489848614` |
| Debias orthogonal mixed | `-1.035898805` | 6 | `8.950830753` | `1.489848614` |

Interpretation:

- The "negative amount clipped away" in one noisy realization is not the same
  object as statistical bias.
- The true bias of a projection/clipping estimator is
  `E[project(true + noise)] - true`, which depends on the unknown true answer.
- Even for simple coordinate-wise clipping at zero, the bias depends on
  `true/sigma`; it cannot be exactly recovered from one noisy draw.
- For the current local-table feasible projection, the situation is more
  coupled:
  - projection is weighted by per-query variances;
  - constraints share mass across overlapping query scopes;
  - lower-bound activity in one query can change neighboring queries;
  - post-projection errors are correlated, while the current bootstrap only
    keeps a diagonal variance approximation.
- Therefore the current plug-in debias can subtract a boundary-bias estimate
  and create negative query targets. QDTE then tries to fit a target outside
  the feasible synthetic-data answer set, which can increase measured residual
  and hurt true RMSE.

Current Judgment:

- Keeping the feasible projected target and only replacing the raw variances by
  projection-aware effective variances is the more defensible mainline right
  now.
- `debias_target=true` should remain an ablation until we have a better
  constrained debias method, such as:
  - partial debias with shrinkage factor `alpha`;
  - debias then reproject;
  - boundary-aware debias that never creates negative targets;
  - offline oracle simulation to estimate actual projection MSE/bias under
    repeated noise draws.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Read current `measurements.json` files and counted negative optimized
  targets for var-only and debias runs.

Current Status:

- The surprising result is explainable: the current debias correction can make
  the optimization target infeasible, while variance-only keeps the target
  feasible and improves weighting.

Next Recommended Task:

- Add an ablation for `debias_alpha in {0.0, 0.25, 0.5, 1.0}` or
  `debias_then_reproject`, and report target infeasibility diagnostics together
  with loss/RMSE.

## Current Method Judgment - 2026-06-18

Task:

- Record the current judgment after clarifying the unweighted-loss scale.

Judgment:

- The `21.9` unweighted loss in the orthogonal variance-only run is not high:
  it corresponds to RMS raw residual `0.4256` counts over `242` measured
  queries.
- The strongest current mainline is:
  - orthogonal grouped measurement/workload construction;
  - feasible consistency projection;
  - projection-aware bootstrap diagonal variance;
  - `debias_target=false`;
  - constructive A-style directed edit generation.
- The main reason is that this setting currently gives the best total offline
  true RMSE in the smoke comparison:
  - raw orthogonal: `0.002338149`;
  - projection-aware variance-only orthogonal: `0.002305222`;
  - projection-aware debias orthogonal: `0.002326633`.
- `debias_target=true` should not be part of the main method yet. It improves
  the orthogonal family fit in one run, but it hurts original mixed badly and
  introduces a large plug-in correction whose stability is not established.
- The orthogonal measurement construction is a real methodological signal:
  it lowers effective noise by measuring query groups with lower sensitivity,
  and the current results suggest it improves true utility, not only measured
  loss.
- The remaining uncertainty is statistical, not conceptual: this is still a
  smoke/single-seed result, so the next decision must come from multi-seed
  comparisons.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation-only update; run `git diff --check` before closing the task.

Current Status:

- Current mainline candidate is selected for follow-up experiments.
- Debias remains an ablation.

Next Recommended Task:

- Run multi-seed experiments comparing raw variance, projection-aware
  variance-only, and projection-aware debias under the orthogonal workload.

## Unweighted Loss Scale Clarification - 2026-06-18

Task:

- Clarify the expected scale of `final_unweighted_measured_loss` after the
  projection-aware variance-only experiment.

Result:

- `unweighted_measured_loss = 0.5 * sum_q residual[q]^2`, without dividing by
  query variance.
- Therefore it is a total over all measured queries, not a per-query number.
- For the current orthogonal variance-only run:
  - queries: `242`;
  - final unweighted loss: `21.917599106`;
  - final RMS raw residual:
    `sqrt(2 * 21.917599106 / 242) = 0.425602141`;
  - final weighted loss: `4.215038075`;
  - effective variance mean: `5.492176549`.
- A one-standard-deviation Gaussian residual field would have:
  - expected weighted loss around `0.5 * 242 = 121`;
  - expected unweighted loss around
    `0.5 * sum_q variance[q] = 664.553362` under the projection-aware
    effective variances.
- So `21.9` unweighted loss is not high in this setting. It means the average
  raw residual to the projected/noisy target is well below one count.

Important Distinction:

- Previous low numbers such as `2-8` were usually weighted measured losses.
- Recent numbers such as `21.9`, `47.0`, `75.9`, or `124.2` are total
  unweighted measured losses, so they are on a different scale.
- Family-level unweighted loss can also be much smaller because it is computed
  over fewer queries. For example the orthogonal-family unweighted loss in the
  variance-only run is `7.194050800` over `44` queries.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Read existing `metrics_final.json`, `metrics_timeseries.csv`, and
  `measurements.json` files from the current smoke outputs.

Current Status:

- Loss-scale interpretation is documented.
- The main comparison should continue to report weighted loss, unweighted loss,
  RMS unweighted residual, and offline true RMSE together.

Next Recommended Task:

- Add a small table/report helper that prints per-query RMS residual and
  one-sigma Gaussian reference scales automatically for future experiments.

## Projection-Aware Variance-Only Ablation - 2026-06-17

Task:

- Run the projection-aware bootstrap experiment again with
  `projection.uncertainty.debias_target=false`.
- Purpose: separate the effect of post-projection variance reweighting from the
  effect of plug-in target debiasing.

Setup:

- Base config: `configs/smoke.yaml`.
- Variant: `constructive_pair`.
- Projection:
  - `projection.consistency.enabled=true`;
  - `projection.consistency.method=local_table_feasible_jax`;
  - `projection.uncertainty.enabled=true`;
  - `projection.uncertainty.method=bootstrap_diagonal`;
  - `projection.uncertainty.num_samples=16`;
  - `projection.uncertainty.center=projected`;
  - `projection.uncertainty.debias_target=false`;
  - `projection.uncertainty.min_raw_variance_fraction=0.05`.
- QDTE:
  - `max_iters=2000`;
  - `stop_patience=2000`;
  - `candidate_diagnostics=false`.

Runs:

- Original mixed:
  `outputs/exp_A_original_mixed_projaware_varonly16_consistency2000_constructive_pair`
- Orthogonal mixed:
  `outputs/exp_A_orthogonal_mixed_projaware_varonly16_consistency2000_constructive_pair`

Result Summary:

| Run | Queries | Final weighted loss | Final unweighted loss | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw original mixed | 243 | `2.318852357` | `75.859848591` | `0.002727697` | 1458 |
| Var-only original mixed | 243 | `8.176284459` | `124.201921078` | `0.002725433` | 1493 |
| Debias original mixed | 243 | `21.724461768` | `338.082179345` | `0.002886751` | 1496 |
| Raw orthogonal mixed | 242 | `3.976314087` | `47.040555620` | `0.002338149` | 942 |
| Var-only orthogonal mixed | 242 | `4.215038075` | `21.917599106` | `0.002305222` | 1033 |
| Debias orthogonal mixed | 242 | `6.047716294` | `28.502779504` | `0.002326633` | 1014 |

Family Results:

| Run | Family | Family queries | Family weighted loss | Family unweighted loss | Family true RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw original | `mixed` | 45 | `1.102442674` | `62.692866258` | `0.004242641` |
| Var-only original | `mixed` | 45 | `4.353294583` | `103.767566613` | `0.004297286` |
| Debias original | `mixed` | 45 | `14.320789789` | `298.303156475` | `0.004756282` |
| Raw orthogonal | `orthogonal_kway_mixed` | 44 | `1.048462780` | `15.726940881` | `0.002969542` |
| Var-only orthogonal | `orthogonal_kway_mixed` | 44 | `1.423972166` | `7.194050800` | `0.002705214` |
| Debias orthogonal | `orthogonal_kway_mixed` | 44 | `1.159312102` | `5.482202860` | `0.002540580` |

Bootstrap Diagnostics:

| Run | Raw variance mean | Effective variance mean | Effective/raw variance mean | Bias L2 | Bias Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original mixed | `19.629629583` | `7.979536403` | `0.497178636` | `14.795448557` | `5.459616661` |
| Orthogonal mixed | `14.504132184` | `5.492176564` | `0.476856508` | `8.950830753` | `1.489848614` |

Interpretation:

- The original mixed degradation mostly comes from target debiasing, not from
  variance reweighting:
  - raw original true RMSE: `0.002727697`;
  - var-only original true RMSE: `0.002725433`;
  - debias original true RMSE: `0.002886751`.
- Orthogonal mixed benefits from projection-aware variance even without
  debiasing:
  - raw orthogonal true RMSE: `0.002338149`;
  - var-only orthogonal true RMSE: `0.002305222`;
  - debias orthogonal true RMSE: `0.002326633`.
- The family-level picture is slightly different:
  - debias gives the best orthogonal-family fit among these three runs
    (`0.002540580` family RMSE);
  - var-only gives the best total true RMSE (`0.002305222`).
- Current best default candidate for the main method is therefore:
  orthogonal workload plus projection-aware bootstrap variance, with
  `debias_target=false`.
- Plug-in debias should remain an ablation, because its estimated bias can be
  large and unstable on overlapping random mixed predicates.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Original mixed variance-only smoke run:
  `scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair ... --projection.uncertainty.debias_target false --qdte.max_iters 2000`
- Orthogonal mixed variance-only smoke run:
  `scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair ... --workload.include_orthogonal_kway_mixed true --projection.uncertainty.debias_target false --qdte.max_iters 2000`
- `git diff --check`
  - passed.

Current Status:

- Projection-aware variance-only is implemented and has been smoke-tested on
  both original mixed and orthogonal mixed workloads.
- The DP boundary is preserved: these optimization runs use projected/noisy
  measurements and variances; true answers are only reported as offline metrics.

Next Recommended Task:

- Run multi-seed comparisons for:
  - raw variance;
  - projection-aware variance-only;
  - projection-aware variance plus debias;
  - original mixed versus orthogonal mixed.
- Treat `debias_target=true` as an exploratory ablation rather than the main
  setting until it is stable across seeds.

## Projection-Aware Experiment Interpretation - 2026-06-17

Task:

- Re-explain the projection-aware bootstrap/debias experiment results in a
  clearer way.

How To Read The Four Runs:

- `raw original mixed`:
  - original random overlapping `mixed` workload;
  - old/raw variance weights after projection;
  - no projection-aware bootstrap variance;
  - no debias target.
- `raw orthogonal mixed`:
  - new `orthogonal_kway_mixed` workload;
  - old/raw variance weights after projection;
  - no projection-aware bootstrap variance;
  - no debias target.
- `projection-aware original mixed`:
  - original random overlapping `mixed` workload;
  - bootstrap-estimated post-projection diagonal variance;
  - plug-in debias target enabled.
- `projection-aware orthogonal mixed`:
  - `orthogonal_kway_mixed` workload;
  - bootstrap-estimated post-projection diagonal variance;
  - plug-in debias target enabled.

Metric Interpretation:

- `weighted loss` uses the run's active weights. It is useful for understanding
  the objective being optimized, but is not directly comparable when we change
  the variance model.
- `unweighted loss` is raw squared residual against that run's measured target.
  It is easier to interpret than weighted loss, but targets can still differ if
  `debias_target=true`.
- `true RMSE` is the most important quality metric in these smoke comparisons.
  It is computed only offline for evaluation.

Simplified Result:

| Run | What changed | True RMSE | Main reading |
| --- | --- | ---: | --- |
| Raw original mixed | baseline workload/weights | `0.002727697` | baseline |
| Raw orthogonal mixed | orthogonal workload only | `0.002338149` | large improvement |
| Projection-aware original mixed | bootstrap variance + debias on original mixed | `0.002886751` | worse than baseline |
| Projection-aware orthogonal mixed | bootstrap variance + debias on orthogonal mixed | `0.002326633` | slightly best total RMSE |

Family-Level Reading:

| Family | Raw unweighted loss | Projection-aware unweighted loss | Raw family true RMSE | Projection-aware family true RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original `mixed` | `62.692866` | `298.303156` | `0.004242641` | `0.004756282` |
| `orthogonal_kway_mixed` | `15.726941` | `5.482203` | `0.002969542` | `0.002540580` |

Interpretation:

- The orthogonal workload itself is robustly helpful:
  - raw original mixed true RMSE: `0.002727697`;
  - raw orthogonal mixed true RMSE: `0.002338149`.
- Projection-aware bootstrap plus debias is not universally helpful:
  - it hurts original random mixed;
  - it slightly helps orthogonal mixed.
- The likely reason is that the plug-in bias estimate is much larger and less
  stable for overlapping random mixed predicates:
  - original mixed `bias_linf ~= 5.46`;
  - orthogonal mixed `bias_linf ~= 1.49`.
- Therefore the immediate conclusion should not be "projection-aware debias is
  solved". The better conclusion is:
  - orthogonal mixed gives a cleaner projected measurement target;
  - projection-aware correction appears compatible with orthogonal mixed;
  - random overlapping mixed is fragile under the same correction.

Important Caveat:

- The current projection-aware run combines two interventions:
  - bootstrap variance replacement;
  - target debiasing.
- Since both were enabled at once, this experiment cannot tell which part
  caused the original mixed degradation. The next ablation must split:
  - `debias_target=false`;
  - `debias_target=true`;
  - different `min_raw_variance_fraction` values.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Not yet run for this documentation-only interpretation.

Next Recommended Task:

- Run projection-aware variance-only experiments, especially
  `projection.uncertainty.debias_target=false`, before deciding whether debias
  should be part of the main method.

## Projection-Aware Bootstrap Variance And Debias - 2026-06-17

Task:

- Implement option 4 from the projection-bias discussion:
  projection-aware post-projection diagonal variance estimation, with optional
  plug-in bias correction.
- Keep the existing raw-variance behavior configurable and unchanged by
  default.

Implementation:

- Added `projection.uncertainty` config:
  - `enabled`: default `false`;
  - `method`: currently `bootstrap_diagonal`;
  - `num_samples`: number of parametric bootstrap samples;
  - `center`: `projected` or `noisy`;
  - `debias_target`: if true, subtracts the estimated projection bias from the
    optimized target;
  - `min_variance`: absolute variance floor;
  - `min_raw_variance_fraction`: optional regularization floor relative to raw
    pre-projection variance.
- Refactored measurement projection into `_apply_configured_projection(...)` so
  the exact same projection operator can be reused by bootstrap samples.
- Bootstrap procedure:
  - choose center, usually `target_projected`;
  - draw `center + N(0, raw_variances)`;
  - re-run the configured projection;
  - estimate diagonal variance from projected bootstrap samples;
  - optionally estimate bias as `mean(projected_bootstrap) - center` and set
    `target_projected := target_projected - bias`.
- This uses only noisy/projected measurements, public variances, public
  workload/schema constraints, and post-processing randomness. It does not use
  exact true answers for optimization, so it respects the DP boundary.

Changed Files:

- `qdte/measurement/measure.py`
- `qdte/config_validation.py`
- `tests/test_measurement.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/measurement/measure.py qdte/config_validation.py tests/test_measurement.py tests/test_config_validation.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m pytest -q tests/test_measurement.py tests/test_config_validation.py`
  - `73 passed in 1.76s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m pytest -q`
  - `167 passed in 10.39s`

Smoke Experiment Setup:

- Base config: `configs/smoke.yaml`.
- Variant: `constructive_pair`.
- Projection:
  - `projection.consistency.enabled=true`;
  - `projection.consistency.method=local_table_feasible_jax`;
  - `projection.uncertainty.enabled=true`;
  - `projection.uncertainty.method=bootstrap_diagonal`;
  - `projection.uncertainty.num_samples=16`;
  - `projection.uncertainty.center=projected`;
  - `projection.uncertainty.debias_target=true`;
  - `projection.uncertainty.min_raw_variance_fraction=0.05`.
- QDTE:
  - `max_iters=2000`;
  - `stop_patience=2000`;
  - `candidate_diagnostics=false`.

Smoke Experiment Results:

| Run | Queries | Final weighted loss | Final unweighted loss | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw original mixed | 243 | `2.318852357` | `75.859848591` | `0.002727697` | 1458 |
| Raw orthogonal mixed | 242 | `3.976314087` | `47.040555620` | `0.002338149` | 942 |
| Projection-aware debias original mixed | 243 | `21.724461768` | `338.082179345` | `0.002886751` | 1496 |
| Projection-aware debias orthogonal mixed | 242 | `6.047716294` | `28.502779504` | `0.002326633` | 1014 |

Family Results:

| Run | Family | Family queries | Family weighted loss | Family unweighted loss | Family true RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw original | `mixed` | 45 | `1.102442674` | `62.692866258` | `0.004242641` |
| Raw orthogonal | `orthogonal_kway_mixed` | 44 | `1.048462780` | `15.726940881` | `0.002969542` |
| Projection-aware original | `mixed` | 45 | `14.320789789` | `298.303156475` | `0.004756282` |
| Projection-aware orthogonal | `orthogonal_kway_mixed` | 44 | `1.159312102` | `5.482202860` | `0.002540580` |

Bootstrap Diagnostics:

| Run | Raw variance mean | Effective variance mean | Effective/raw variance mean | Bias L2 | Bias Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projection-aware original | `19.629629583` | `7.979536403` | `0.497178636` | `14.795448557` | `5.459616661` |
| Projection-aware orthogonal | `14.504132184` | `5.492176564` | `0.476856508` | `8.950830753` | `1.489848614` |

Interpretation:

- The implemented option 4 is functional and passes tests.
- Naive plug-in debias plus bootstrap variance is not uniformly better:
  - it worsens original random mixed in both true RMSE and unweighted measured
    loss;
  - it slightly improves orthogonal mixed total true RMSE
    (`0.002338149 -> 0.002326633`) and strongly improves orthogonal family
    unweighted loss (`15.726940881 -> 5.482202860`).
- The debias estimate is much larger for original mixed (`bias_linf ~= 5.46`)
  than for orthogonal mixed (`bias_linf ~= 1.49`), which suggests the plug-in
  bias correction may be unstable for overlapping random mixed predicates.
- The orthogonal workload remains the cleaner story: projection-aware
  uncertainty strengthens its family-level fit, while random mixed becomes more
  fragile under the same correction.
- Performance caveat: the current bootstrap implementation reruns the full
  configured projection per sample. With `local_table_feasible_jax` and
  `num_samples=16`, startup cost is noticeably high. This is correct but not
  yet optimized.

Current Status:

- Option 4 exists as configurable code and has a first smoke comparison.
- Default behavior remains unchanged unless `projection.uncertainty.enabled`
  is set.

Next Recommended Task:

- Split the ablation into:
  - bootstrap variance only, `debias_target=false`;
  - bootstrap variance plus debias, `debias_target=true`;
  - lower/higher `min_raw_variance_fraction`;
  - raw variance and unweighted scoring baselines.
- If projection-aware uncertainty remains promising, optimize performance by
  caching projection matrices/constraints or implementing active-set
  linearized covariance for `local_table_feasible_jax`.

## Projection Bias And Weighted Objective Caveat - 2026-06-17

Task:

- Clarify whether inverse-variance weighting remains statistically correct
  after consistency/feasibility projection.

Key Point:

- The clean weighted least-squares interpretation assumes a measurement model:
  - `target = true_answer + zero_mean_noise`;
  - known covariance/variance;
  - residuals are standardized by that uncertainty.
- After projection, the target is:
  - `target_projected = projection(target_noisy)`.
- If the projection is a linear equality projection onto a constraint subspace
  containing the true answers, then it can remain unbiased:
  - `E[P(target_noisy)] = P(true_answer) = true_answer`;
  - but the covariance becomes `P Sigma P^T`, generally correlated and not the
    original diagonal variance.
- If the projection includes nonlinear operations such as clipping,
  non-negativity, or active-set feasibility constraints, then
  `target_projected` can be biased:
  - `E[projection(true_answer + noise)] != true_answer`;
  - the bias is strongest near active boundaries such as zero cells.

Implication:

- The current objective is best interpreted as a quasi-likelihood /
  confidence-weighted residual objective against the projected target, not a
  fully exact likelihood for the post-projection estimator.
- Simply "subtracting the mean bias" is not currently available because the
  bias depends on the unknown true answer and on which constraints become
  active. Estimating it from exact true answers would violate the DP boundary.
- A more statistically complete post-projection objective would need one of:
  - local active-set linearization to estimate post-projection covariance;
  - bootstrap/simulation using only noisy/projected measurements to approximate
    projection bias and covariance;
  - a full constrained measurement model that treats projection as part of the
    likelihood.

Current Status:

- Current code still uses the original per-query variances as diagonal weights
  after projection.
- This is defensible as an optimization heuristic and confidence weighting, but
  should not be oversold as an exact post-projection likelihood when nonlinear
  feasible projection is enabled.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- `git diff --check`

Next Recommended Task:

- Add a projection-aware weighting ablation:
  - current raw inverse variance;
  - unweighted;
  - optionally local active-set covariance/diagonal variance if exposed by the
    projection solver.

## Why Weighted Loss Divides By Noise Variance - 2026-06-17

Task:

- Clarify why the measured objective divides squared residuals by noise
  variance, and why this does not mean larger noise is better.

Explanation:

- The DP measurement target is a noisy observation, not ground truth:
  - `y[q] = true_answer[q] + noise[q]`;
  - `noise[q] ~ N(0, variance[q])`.
- If a synthetic answer `mu[q] = answer_syn[q]` is evaluated against this
  observation under the Gaussian measurement model, the negative log likelihood
  contains:
  - `0.5 * (y[q] - mu[q])^2 / variance[q]`;
  - plus `0.5 * log(variance[q])`, which is constant with respect to `mu[q]`
    during a fixed QDTE run.
- QDTE therefore uses the weighted least-squares part:
  - `measured_loss = 0.5 * sum_q residual[q]^2 / variance[q]`.
- Intuition:
  - the loss penalizes residuals in units of standard deviations, not raw count
    units;
  - a residual of `5` is severe if `noise_std=1`;
  - a residual of `5` is weak evidence if `noise_std=50`.
- This prevents QDTE from overfitting highly noisy measurements. High-noise
  queries are less reliable directional signals, so their residuals should have
  less influence on active-query selection and edit scoring.

Important Caveat:

- Within one fixed run, the optimizer cannot make noise larger to reduce loss;
  variances are fixed by the DP measurement mechanism before optimization.
- Across different measurement designs, weighted measured loss is not a pure
  quality metric. A high-noise design can have smaller weighted residual terms
  for the same raw residual, while also containing less useful information.
- For cross-design comparison, report:
  - weighted measured loss, to understand the objective being optimized;
  - unweighted measured loss, to inspect raw residual magnitude;
  - offline true MAE/RMSE or common heldout true RMSE, to judge synthetic data
    quality.
- If one wants a full likelihood comparison across designs with different
  variances, the omitted `0.5 * log(variance[q])` constants matter. They are
  irrelevant for edit ranking within a fixed run but matter conceptually when
  comparing noise mechanisms.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Not yet run for this documentation-only clarification.

Next Recommended Task:

- When presenting experiments, explicitly label weighted measured loss as a
  standardized residual objective, not as raw fitting error. Pair it with
  unweighted measured loss and heldout true error.

## Family Loss And Orthogonal Workload Innovation Note - 2026-06-17

Task:

- Clarify the meaning of family loss.
- Explain why unweighted measured loss is much larger than weighted measured
  loss.
- Record that the orthogonal mixed family improvement is a candidate
  innovation point.

Explanation:

- `family loss` is not a new objective. It is the same residual loss computed
  on a subset of measured queries grouped by workload family:
  - `oneway`;
  - `twoway`;
  - `prefix`;
  - `range`;
  - `mixed`;
  - `orthogonal_kway_mixed`.
- For a family query index set `F`:
  - weighted family loss is
    `0.5 * sum_{q in F} residual[q]^2 * inv_variance[q]`;
  - unweighted family loss is
    `0.5 * sum_{q in F} residual[q]^2`;
  - family true RMSE is an offline evaluation metric against exact true
    answers, and is not used by DP optimization.
- Family losses decompose where the remaining residual comes from. They are
  useful because total loss can hide that one workload family improved while
  another family dominates the aggregate.

Why Unweighted Loss Is Much Larger:

- The current weighted loss divides squared residuals by the noise variance:
  - `0.5 * residual[q]^2 / variance[q]`.
- In the smoke DP setup, common noise variances are much larger than `1`:
  - oneway/twoway `noise_std ~= 3.162`, variance `~= 10`;
  - prefix/range `noise_std ~= 4.47` to `7.30`, variance `~= 20` to `53`;
  - original mixed `noise_std ~= 5.48` to `8.66`, variance `~= 30` to `75`;
  - orthogonal mixed `noise_std ~= 3.873`, variance `~= 15`.
- Therefore weighted loss is often one order of magnitude smaller than
  unweighted loss because each squared count residual is scaled by roughly
  `1/10` to `1/75`.
- This is a unit change: unweighted loss measures raw count residuals, while
  weighted loss measures standardized residuals in noise units.

Innovation Note:

- The matched-count diagnostic shows that orthogonal mixed is especially strong
  at the family level:
  - original `mixed` final unweighted family loss: `62.692866258`;
  - `orthogonal_kway_mixed` final unweighted family loss: `15.726940881`;
  - original `mixed` family true RMSE: `0.004242641`;
  - `orthogonal_kway_mixed` family true RMSE: `0.002969542`.
- This should not be lost in the paper narrative. It supports a separate
  measurement/workload-design contribution:
  - construct orthogonal mixed query groups, especially disjoint range bins, so
    group sensitivity stays `1`;
  - lower measurement noise improves the usable projected target;
  - QDTE then has a cleaner directional residual signal for evolution.
- The caveat is that original random `mixed` predicates and
  `orthogonal_kway_mixed` partition predicates are not identical query sets, so
  final claims should include:
  - matched query-count comparisons;
  - common heldout workload evaluation;
  - family-level weighted and unweighted diagnostics.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Not yet run for this documentation-only clarification.

Next Recommended Task:

- In the experiment summary/report table, always include total metrics and
  family-level metrics for measured-loss variants:
  - weighted measured loss;
  - unweighted measured loss;
  - RMS unweighted residual;
  - offline true RMSE.

## Matched A Unweighted Loss Diagnostic Run - 2026-06-17

Task:

- Re-run the matched original mixed vs orthogonal mixed A-scheme comparison
  after adding unweighted measured-loss diagnostics.
- Check whether orthogonal mixed's higher weighted measured loss is caused by
  stricter inverse-variance weighting rather than larger absolute residuals.

Setup:

- Base config: `configs/smoke.yaml`.
- Variant: `constructive_pair`.
- Projection:
  - `projection.consistency.enabled=true`;
  - `projection.consistency.method=local_table_feasible_jax`;
  - `projection.consistency.max_scope_cells=200000`;
  - `projection.consistency.max_dense_constraint_cells=20000000`.
- QDTE:
  - `max_iters=2000`;
  - `stop_patience=2000`;
  - `log_every=200`;
  - `candidate_diagnostics=false`.

Runs:

- Original mixed:
  - output:
    `outputs/exp_A_matched45_original_mixed_unweighteddiag_consistency2000_constructive_pair/`;
  - total queries: `243`;
  - mixed family queries: `45`.
- Orthogonal mixed:
  - output:
    `outputs/exp_A_matched45_orthogonal_mixed_unweighteddiag_consistency2000_constructive_pair/`;
  - total queries: `242`;
  - orthogonal mixed family queries: `44`;
  - `orthogonal_kway_mixed_range_bins=2`;
  - `orthogonal_kway_mixed_scopes_per_order=6`.

Total Results:

| Run | Queries | Final weighted loss | Final unweighted loss | RMS standardized residual | RMS unweighted residual | Final true RMSE | Candidates scored | Accepted edits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original mixed | 243 | `2.318852357` | `75.859848591` | `0.138149211` | `0.790165105` | `0.002727697` | 512000 | 1458 |
| Orthogonal mixed | 242 | `3.976314087` | `47.040555620` | `0.181279066` | `0.623510162` | `0.002338149` | 35072 | 942 |

Mixed-Family Results:

| Run | Family queries | Final weighted family loss | Final unweighted family loss | Family RMS unweighted residual | Family true RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original `mixed` | 45 | `1.102442674` | `62.692866258` | `1.669236236` | `0.004242641` |
| `orthogonal_kway_mixed` | 44 | `1.048462780` | `15.726940881` | `0.845494500` | `0.002969542` |

Interpretation:

- The total weighted loss is higher for the orthogonal mixed run:
  - `2.318852357 -> 3.976314087`.
- The total unweighted measured residual loss is lower for the orthogonal mixed
  run:
  - `75.859848591 -> 47.040555620`.
- Offline true RMSE is also lower:
  - `0.002727697 -> 0.002338149`.
- Therefore this run supports the hypothesis that the higher total weighted
  loss mostly reflects stricter inverse-variance weighting / standardized
  residuals, not worse absolute residual fit.
- At the mixed-family level, orthogonal mixed is better under both weighted and
  unweighted measured loss, but the family predicates are not identical between
  original random mixed and orthogonal partition mixed.

Weighting Note:

- `inv_variance = 1 / variance` is a model-based default from Gaussian
  measurement likelihood / weighted least squares, not an arbitrary constant.
- It can still be studied as an algorithmic design choice. A natural ablation
  is `weight[q] = inv_variance[q]^alpha`, where:
  - `alpha=1` is the current statistically calibrated objective;
  - `alpha=0` is unweighted residual optimization;
  - `0 < alpha < 1` tempers aggressive variance weighting.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Experiment commands:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_A_matched45_original_mixed_unweighteddiag_consistency2000 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 200 --qdte.candidate_diagnostics false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_A_matched45_orthogonal_mixed_unweighteddiag_consistency2000 --workload.include_mixed false --workload.include_orthogonal_kway_mixed true --workload.orthogonal_kway_mixed_orders 2 --workload.orthogonal_kway_mixed_scopes_per_order 6 --workload.orthogonal_kway_mixed_range_bins 2 --workload.max_queries 400 --privacy.measurement_allocation.orthogonal_kway_mixed 0.20 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 200 --qdte.candidate_diagnostics false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`

Next Recommended Task:

- Add a scoring-weight tempering ablation, e.g. `score_weight_alpha`, and run
  `alpha in {0, 0.5, 1}` to test whether calibrated weighting or tempered
  weighting gives better true heldout RMSE for QDTE.

## Unweighted Measured Loss Diagnostics - 2026-06-17

Task:

- Add an unweighted measured-loss diagnostic so experiments can show whether a
  higher weighted measured loss is caused by larger absolute residuals or by
  stricter inverse-variance weights.
- Clarify that QDTE is not using backpropagation, but `edit_advantage` is the
  closed-form one-step reduction of the weighted residual loss.

Implementation:

- Added metric helpers:
  - `unweighted_measured_loss(residual) = 0.5 * sum_q residual[q]^2`;
  - `rms_unweighted_residual(residual) = sqrt(mean_q residual[q]^2)`.
- Added final metrics:
  - `initial_unweighted_measured_loss`;
  - `final_unweighted_measured_loss`;
  - `unweighted_measured_loss_reduction`;
  - `initial_rms_unweighted_residual`;
  - `final_rms_unweighted_residual`.
- Added timeseries columns:
  - `unweighted_measured_loss`;
  - `rms_unweighted_residual`.
- Added family-level unweighted loss and RMS residual fields.
- Added logs:
  - per-logged-iteration `unweighted_loss=...`;
  - final `Final unweighted measured loss: ...`.
- Optimization behavior is unchanged. Candidate scoring still uses the original
  weighted `edit_advantage` objective.

Interpretation:

- For a candidate edit, `answer_syn` changes by `delta`, so residual changes
  from `r` to `r - delta`.
- The weighted loss reduction is:
  - `0.5 * sum_q (r[q]^2 - (r[q] - delta[q])^2) * inv_variance[q]`;
  - equivalently
    `delta @ (residual * inv_variance) - 0.5 * ((delta * delta) @ inv_variance)`.
- This is exactly the current `edit_advantage` before subtracting
  `lambda_cost * edit_cost`.
- Therefore the loss definition matters even without gradients: it determines
  which discrete edit has the highest expected objective decrease.

Changed Files:

- `qdte/eval/metrics.py`
- `qdte/evolution/engine.py`
- `tests/test_metrics.py`
- `tests/test_engine_smoke.py`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/eval/metrics.py qdte/evolution/engine.py tests/test_metrics.py tests/test_engine_smoke.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m pytest -q tests/test_metrics.py tests/test_engine_smoke.py`
  - `7 passed in 3.60s`
- `git diff --check`

Next Recommended Task:

- Re-run the matched original mixed vs orthogonal mixed A-scheme experiment and
  compare both `final_measured_loss` and `final_unweighted_measured_loss`.
- If needed, add a separate ablation where scoring uses unweighted
  `inv_variance = 1` to test whether weighted scoring itself changes
  optimization quality.

## Weighted Loss Affects Discrete Optimization - 2026-06-17

Task:

- Clarify whether the loss definition matters when QDTE is not trained by
  backpropagation.

Explanation:

- QDTE is not a gradient/backpropagation method, but the measured objective
  still affects optimization directly through the discrete edit score.
- Candidate edits are ranked by edit advantage:
  - `delta[q] = phi_q(x_new) - phi_q(x_old)`;
  - `edit advantage = delta @ (residual * inv_variance) - 0.5 * ((delta * delta) @ inv_variance) - lambda_cost * edit_cost`.
- This formula is exactly the one-step decrease of the weighted measured loss
  up to edit cost. Therefore changing `inv_variance` changes which candidate
  edits look beneficial.
- Noise scale also affects active query selection because active priorities use
  standardized residuals such as `abs(residual) / sigma` and noise-threshold
  tests such as `abs(residual) > kappa_noise * sigma`.
- Therefore the weighted loss is not just a reporting metric. It participates
  in:
  - active query selection;
  - candidate generation for residual/weighted compilers;
  - candidate scoring and acceptance;
  - query debt/collateral-damage accounting;
  - stopping behavior when no query is above the noise threshold.
- An unweighted objective is possible as an ablation, but it would be a
  different algorithmic choice: it would treat noisy high-variance measurements
  as equally reliable as low-variance measurements.

Current Status:

- This clarification is now documented.
- No source code changed.

Tests Run:

- `git diff --check`

Next Recommended Task:

- If this remains uncertain, add an ablation that scores edits with
  `inv_variance = 1` while keeping the DP target fixed, then compare true
  heldout RMSE and measured loss under the original weighted metric.

## Measured Loss And Noise Variance Interpretation - 2026-06-17

Task:

- Clarify whether and how DP measurement noise affects QDTE measured loss.

Explanation:

- QDTE measured loss is:
  - `residual[q] = target_projected[q] - answer_syn[q]`;
  - `measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]`.
- The inverse-variance weight is used because the noisy/projected target is a
  measurement with known uncertainty, not an exact answer. Under independent
  Gaussian measurement noise, this is the weighted least-squares / maximum
  likelihood objective:
  - `0.5 * (target_projected[q] - answer_syn[q])^2 / variance[q]`.
- Using an unweighted squared loss would implicitly assume all query
  measurements have equal variance. That is only equivalent when all variances
  are the same up to a constant.
- Intuitively, the weighted loss measures residuals in units of noise standard
  deviations:
  - a residual of `5` against `noise_std=10` is weak evidence of mismatch;
  - a residual of `5` against `noise_std=1` is strong evidence of mismatch.
- DP noise affects `measured_loss` in two ways:
  - it changes the optimized target through `target_noisy` and
    `target_projected`;
  - it changes the per-query weight through
    `inv_variance[q] = 1 / variance[q]`.
- Lower-sensitivity group measurement has smaller noise variance, so the same
  absolute residual is penalized more heavily in measured loss.
- Therefore measured losses from different measurement designs are not always
  directly comparable unless the query set and variances are comparable.
- Offline true MAE/RMSE are computed only for evaluation and are the better
  way to compare whether a new measurement scheme improves actual synthetic
  quality in DP experiments.

Current Status:

- This interpretation is now documented.
- No source code changed.

Tests Run:

- Not run; documentation-only clarification.

Next Recommended Task:

- For future measurement-scheme comparisons, report both weighted measured
  loss and unweighted true/target residual metrics, preferably on a common
  held-out query set.

## Matched-Count A Scheme Orthogonal Mixed Pilot - 2026-06-17

Task:

- Re-run the A scheme comparison after matching the number of mixed-like
  measured queries.
- The concern was that the previous orthogonal mixed run used `88` mixed-like
  queries versus `45` original mixed queries, which could itself provide more
  information.

Setup:

- Base config: `configs/smoke.yaml`.
- Variant: `constructive_pair`.
- Projection:
  - `projection.consistency.enabled=true`;
  - `projection.consistency.method=local_table_feasible_jax`;
  - `projection.consistency.max_scope_cells=200000`;
  - `projection.consistency.max_dense_constraint_cells=20000000`.
- QDTE:
  - `max_iters=2000`;
  - `stop_patience=2000`;
  - `total_candidates_per_iter=256`;
  - `accepted_per_iter=8`;
  - `candidate_diagnostics=false`.

Matched Runs:

- Original random mixed:
  - output: `outputs/exp_A_matched45_original_mixed_consistency2000_constructive_pair/`;
  - total queries: `243`;
  - mixed-like queries: `45`;
  - mixed group sensitivity: `sqrt(2)` to `sqrt(5)`;
  - mixed group noise std: `5.477` to `8.660`.
- Orthogonal mixed:
  - output: `outputs/exp_A_matched45_orthogonal_mixed_consistency2000_constructive_pair/`;
  - total queries: `242`;
  - mixed-like queries: `44`;
  - used `orthogonal_kway_mixed_range_bins=2`;
  - all orthogonal mixed group sensitivities: `1`;
  - orthogonal mixed group noise std: `3.873`.

Result:

| Run | Total queries | Mixed-like queries | Final measured loss | Final true MAE | Final true RMSE | Candidates scored | Accepted edits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A + original random mixed | 243 | 45 | `2.278054872` | `0.002020576` | `0.002725433` | 512000 | 1456 |
| A + orthogonal mixed | 242 | 44 | `3.976314087` | `0.001805785` | `0.002338149` | 35072 | 942 |

Family-level result:

| Run | Family | Family queries | Final family measured loss | Final family true RMSE |
| --- | --- | ---: | ---: | ---: |
| Original random mixed | `mixed` | 45 | `1.118008105` | `0.004234777` |
| Orthogonal mixed | `orthogonal_kway_mixed` | 44 | `1.048462780` | `0.002969542` |

Interpretation:

- After matching query counts, the orthogonal mixed measurement still improves
  offline true error on this seed:
  - total final true RMSE: `0.002725433 -> 0.002338149`;
  - total final true MAE: `0.002020576 -> 0.001805785`.
- The mixed-like family itself also improves in true RMSE:
  - `0.004234777 -> 0.002969542`.
- The orthogonal run's measured loss is higher despite lower true RMSE because
  the lower-noise orthogonal measurements have larger `inv_variance`, so the
  measured objective is stricter. It also stopped generating candidates after
  residuals fell below the configured noise threshold, scoring only `35072`
  candidates versus `512000` for the original mixed run.
- This is still a one-seed pilot and the measured mixed predicates are not
  identical predicates, but the query-count confound is substantially reduced.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- No unit tests were run in this task because no source code changed.
- Experiment commands:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_A_matched45_original_mixed_consistency2000 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 200 --qdte.candidate_diagnostics false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_A_matched45_orthogonal_mixed_consistency2000 --workload.include_mixed false --workload.include_orthogonal_kway_mixed true --workload.orthogonal_kway_mixed_orders 2 --workload.orthogonal_kway_mixed_scopes_per_order 6 --workload.orthogonal_kway_mixed_range_bins 2 --workload.max_queries 400 --privacy.measurement_allocation.orthogonal_kway_mixed 0.20 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 200 --qdte.candidate_diagnostics false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`

Current Status:

- The query-count-matched pilot continues to support the orthogonal mixed
  measurement scheme.
- Exact `45` orthogonal mixed queries would require truncating a partition or
  adding a partial group; this run kept complete partitions, yielding `44`
  mixed-like queries.

Next Recommended Task:

- Run a multi-seed matched-count comparison and evaluate both methods on a
  common held-out workload, so the final claim is not tied to different
  measured query predicates.

## A Scheme With Orthogonal Mixed Measurement Pilot - 2026-06-17

Task:

- Test whether the lower-sensitivity orthogonal mixed measurement scheme helps
  downstream true error after consistency projection.
- Run the original A scheme, i.e. `run_ablation.py --variant constructive_pair`.

Setup:

- Base config: `configs/smoke.yaml`.
- Variant: `constructive_pair`.
- Projection:
  - `projection.consistency.enabled=true`;
  - `projection.consistency.method=local_table_feasible_jax`;
  - `projection.consistency.max_scope_cells=200000`;
  - `projection.consistency.max_dense_constraint_cells=20000000`.
- QDTE:
  - `max_iters=2000`;
  - `stop_patience=2000`;
  - `total_candidates_per_iter=256`;
  - `accepted_per_iter=8`;
  - `candidate_diagnostics=false`.

Runs:

- Orthogonal mixed measurement:
  - output: `outputs/exp_A_orthogonal_mixed_consistency2000_constructive_pair/`;
  - disabled random `mixed`;
  - enabled `orthogonal_kway_mixed`;
  - `orthogonal_kway_mixed_orders=2`;
  - `orthogonal_kway_mixed_scopes_per_order=6`;
  - `orthogonal_kway_mixed_range_bins=4`;
  - `privacy.measurement_allocation.orthogonal_kway_mixed=0.20`.
- Current-code original mixed baseline:
  - output: `outputs/exp_A_original_mixed_consistency2000_constructive_pair/`;
  - original random `mixed` workload;
  - exact group-sensitivity refinement enabled by current builder.

Results:

| Run | Queries | Mixed-like family | Mixed-like queries | Mixed-like sensitivity | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| A + original random mixed | 243 | `mixed` | 45 | `sqrt(2)` to `sqrt(5)` | `2.278054872` | `0.002020576` | `0.002725433` | 1456 |
| A + orthogonal mixed | 286 | `orthogonal_kway_mixed` | 88 | `1` for every mixed group | `2.850210873` | `0.001723776` | `0.002239974` | 1235 |

Family-level observation:

- Original random `mixed` final true RMSE: `0.004234777`.
- Orthogonal mixed family final true RMSE: `0.002402650`.
- These two family numbers are not exactly apples-to-apples because the query
  set changed from random overlapping predicates to Cartesian partition cells,
  but the direction is favorable.

Interpretation:

- On this seed and smoke setup, the lower-sensitivity orthogonal mixed
  measurement plus consistency projection improves final offline true error:
  - total final true RMSE improves from `0.002725433` to `0.002239974`;
  - total final true MAE improves from `0.002020576` to `0.001723776`.
- The final measured loss is lower for the original mixed run, but it is not a
  clean comparison because the workload query set and query count differ.
- The mixed group noise scale confirms the intended mechanism:
  - original random mixed groups have sensitivity between `sqrt(2)` and
    `sqrt(5)`, with noise std `5.477` to `8.660`;
  - orthogonal mixed groups all have sensitivity `1`, with noise std `3.873`
    under the same per-group rho.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- No unit tests were run in this task because no source code changed.
- Experiment commands:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_A_orthogonal_mixed_consistency2000 --workload.include_mixed false --workload.include_orthogonal_kway_mixed true --workload.orthogonal_kway_mixed_orders 2 --workload.orthogonal_kway_mixed_scopes_per_order 6 --workload.orthogonal_kway_mixed_range_bins 4 --workload.max_queries 400 --privacy.measurement_allocation.orthogonal_kway_mixed 0.20 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 200 --qdte.candidate_diagnostics false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_A_original_mixed_consistency2000 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 200 --qdte.candidate_diagnostics false --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`

Current Status:

- The pilot supports the hypothesis that orthogonal mixed measurement can
  reduce downstream true error after consistency projection.
- The result is one seed and not a fully fair workload-matched experiment,
  because the random mixed and orthogonal mixed query sets differ.

Next Recommended Task:

- Run a matched multi-seed ablation where evaluation includes a common held-out
  query set for both measurement schemes.
- Consider adding an orthogonal standalone range workload so range and mixed
  measurement can both exploit partition sensitivity.

## Orthogonal K-Way Mixed Workload Construction - 2026-06-17

Task:

- Add an orthogonal query construction scheme for mixed equality/range
  workloads.
- Keep the existing random `mixed` and `kway_mixed` generators unchanged for
  ablation comparisons.
- Clarify why orthogonal halfspace groups require a query-representation
  extension.

Implemented:

- Added `include_orthogonal_kway_mixed`.
  - Family: `orthogonal_kway_mixed`.
  - Samples k-way mixed scopes containing at least one categorical and one
    numerical attribute.
  - Categorical attributes are expanded into all equality cells.
  - Numerical attributes are expanded into equal-width disjoint `OP_RANGE`
    intervals.
  - The Cartesian product of those cells is a mutually exclusive and complete
    partition over the selected scope.
  - Each generated `WorkloadGroup` has `sensitivity_l2=1` and
    `is_partition=True`.
- Added controls:
  - `orthogonal_kway_mixed_orders`, default `[2]`;
  - `orthogonal_kway_mixed_scopes_per_order`, default `4` in ordinary workload
    summary and `16` in held-out defaults;
  - `orthogonal_kway_mixed_range_bins`, default `4`;
  - `orthogonal_kway_mixed_max_cells_per_group`, default `4096`.
- Added the new family to workload summaries and held-out workload defaults.
- Added regression coverage proving a 2-way categorical/range partition creates
  four disjoint queries with group sensitivity `1`.

DP/Noise Rationale:

- For a group answer vector, the relevant sensitivity is:
  `sqrt(max_x number_of_group_queries_satisfied_by_x)`.
- The orthogonal construction makes that maximum equal to `1`.
- This means a whole equality/range partition can be measured as a vector with
  the same zCDP Gaussian noise scale as one scalar query.

Halfspace Status:

- Orthogonal halfspace groups are not implemented in this change.
- The current `QueryCatalogue` represents one cumulative halfspace predicate:
  `sum_i w_i x_i <= t`.
- Equal-width thresholds over that score are still nested/cumulative queries,
  not mutually exclusive queries.
- A true sensitivity-1 halfspace partition needs either:
  - a linear slab predicate, e.g. `lo < sum_i w_i x_i <= hi`; or
  - halfspace-tree leaf predicates, i.e. conjunctions of halfspace decisions.
- Either path requires extending query representation, JAX evaluation, delta
  scoring, GPU repair/scoring, and possibly consistency projection.

Changed Files:

- `qdte/queries/workload.py`
- `qdte/evolution/engine.py`
- `tests/test_workload.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/queries/workload.py qdte/evolution/engine.py tests/test_workload.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_workload.py tests/test_measurement.py`
  - `18 passed in 1.51s`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_orthogonal_kway_mixed_probe --workload.include_oneway false --workload.include_2way_cat false --workload.include_prefix false --workload.include_range false --workload.include_mixed false --workload.include_kway false --workload.include_kway_prefix false --workload.include_kway_range false --workload.include_kway_mixed false --workload.include_orthogonal_kway_mixed true --workload.orthogonal_kway_mixed_orders 2 --workload.orthogonal_kway_mixed_scopes_per_order 1 --workload.orthogonal_kway_mixed_range_bins 4 --workload.max_queries 64 --privacy.measurement_allocation.orthogonal_kway_mixed 1.0 --qdte.max_iters 3 --qdte.stop_patience 3 --qdte.log_every 1 --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - passed
  - workload: `20` queries, `1` `orthogonal_kway_mixed` group
  - measurement group: sensitivity `1`, rho `1`, noise std `0.707107`
  - final measured loss: `93636.5`
  - candidates scored: `768`
  - accepted edits: `24`

Current Status:

- The code now has both random overlapping mixed workloads and an orthogonal
  equality/range mixed workload construction.
- The orthogonal equality/range construction can be used to test whether
  lower group sensitivity improves total query error and downstream QDTE
  convergence.

Next Recommended Task:

- Run a controlled ablation comparing `kway_mixed` versus
  `orthogonal_kway_mixed` at matched scope/order/query budgets.
- Design the halfspace-slab representation if orthogonal halfspace partitions
  remain important for the paper.

## Exact Group Sensitivity For Orthogonal Workloads - 2026-06-17

Task:

- Correct the DP sensitivity used when grouped queries are orthogonal.
- Explain why the previous `kway_mixed` smoke reported sensitivity `4`.

Implemented:

- Added exact `WorkloadGroup` L2 sensitivity refinement in
  `qdte/queries/workload.py`.
- The refinement enumerates the finite public query scope and computes:
  - `max_overlap = max_x number_of_group_queries_satisfied_by_x`;
  - `sensitivity_l2 = sqrt(max_overlap)`.
- This uses only schema/cardinality and query definitions, not private data.
- It is controlled by `workload.exact_group_sensitivity_max_cells`, default
  `200000`.
- If the group scope is too large to enumerate, the builder keeps the previous
  conservative sensitivity.
- Orthogonal equality workloads now get sensitivity `1`, as expected.
- Overlapping prefix/range/mixed workloads get their actual maximum overlap
  when the scope is enumerable.

Important Correction:

- The earlier `kway_mixed` smoke reported sensitivity `4` because non-partition
  groups were initialized with the conservative `sqrt(group_size)` bound.
- That was safe but too noisy, and it did not exploit orthogonality.
- Re-running the same 16-query `kway_mixed` smoke after exact refinement gives
  sensitivity `1.41421`, because the sampled mixed predicates are not fully
  orthogonal but have maximum overlap `2`.
- A same-scope k-way equality cell workload now has sensitivity `1`.

Changed Files:

- `qdte/queries/workload.py`
- `tests/test_workload.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/queries/workload.py tests/test_workload.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_workload.py tests/test_measurement.py`
  - `17 passed in 1.49s`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_kway_mixed_sensitivity_probe --workload.include_oneway false --workload.include_2way_cat false --workload.include_prefix false --workload.include_range false --workload.include_mixed false --workload.include_kway false --workload.include_kway_prefix false --workload.include_kway_range false --workload.include_kway_mixed true --workload.kway_mixed_orders 4 --workload.kway_mixed_queries_per_order 16 --workload.max_queries 32 --privacy.measurement_allocation.kway_mixed 1.0 --qdte.max_iters 3 --qdte.stop_patience 3 --qdte.log_every 1 --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - passed
  - workload: `16` queries, `1` `kway_mixed` group
  - measurement group: sensitivity `1.41421`, rho `1`, noise std `1`
  - final measured loss: `9174.93`
  - candidates scored: `768`
  - accepted edits: `24`

Current Status:

- Grouped DP measurement now exploits orthogonality when the group query scope
  is enumerable.
- The previous `sqrt(group_size)` behavior remains only as a fallback for
  too-large scopes.

Next Recommended Task:

- Decide whether high-order sampled workloads should be grouped by exact scope
  to expose more orthogonal subgroups and possibly improve the budget/noise
  tradeoff.

## K-Way Mixed Workloads And Group Measurement - 2026-06-17

Task:

- Answer whether `k-way-mixed` and grouped query noise are implemented.
- Implement missing `kway_mixed` workload support and document the actual
  group-noise semantics.

Implemented:

- Added `include_kway_mixed`.
  - Family: `kway_mixed`.
  - Generates sampled k-way conjunctions.
  - Categorical attributes use `OP_EQ`.
  - Numerical attributes use randomly sampled `OP_LE` prefix or `OP_RANGE`
    range terms.
  - Requires at least one numerical attribute in each sampled query and allows
    multiple numerical terms in the same query.
  - Controlled by `kway_mixed_orders` and
    `kway_mixed_queries_per_order`.
- Extended workload summaries and held-out workload defaults so
  `kway_mixed` controls are preserved in run artifacts.
- Added regression coverage for `kway_mixed` query generation.
- Added regression coverage for grouped DP measurement calibration.

Group Measurement Status:

- The static DP measurement path already measures per `WorkloadGroup`.
- For each group, `measure_real_dataset` computes the group answer vector and
  calls the zCDP Gaussian mechanism calibrated by:
  - the group's `sensitivity_l2`;
  - the rho allocated to that group's family, divided across groups in that
    family.
- Current semantics are independent Gaussian coordinates with the same
  standard deviation inside a group, and diagonal variance bookkeeping.
- This is not adaptive select-measure-generate yet; `measurement_mode` is still
  `static_all`.

Changed Files:

- `qdte/queries/workload.py`
- `qdte/evolution/engine.py`
- `tests/test_workload.py`
- `tests/test_measurement.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/queries/workload.py qdte/evolution/engine.py tests/test_workload.py tests/test_measurement.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_workload.py tests/test_measurement.py`
  - `16 passed in 1.47s`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_kway_mixed_workload_probe --workload.include_oneway false --workload.include_2way_cat false --workload.include_prefix false --workload.include_range false --workload.include_mixed false --workload.include_kway false --workload.include_kway_prefix false --workload.include_kway_range false --workload.include_kway_mixed true --workload.kway_mixed_orders 4 --workload.kway_mixed_queries_per_order 16 --workload.max_queries 32 --privacy.measurement_allocation.kway_mixed 1.0 --qdte.max_iters 3 --qdte.stop_patience 3 --qdte.log_every 1 --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - passed
  - workload: `16` queries, `1` `kway_mixed` group
  - measurement group: sensitivity `4`, rho `1`, noise std `2.82843`
  - final measured loss: `1146.75`
  - candidates scored: `768`
  - accepted edits: `24`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `160 passed in 11.76s`
- `git diff --check`
  - passed

Current Status:

- `kway`, `kway_prefix`, `kway_range`, and `kway_mixed` are implemented and
  disabled by default.
- Enabling `kway_mixed` in DP mode requires
  `privacy.measurement_allocation.kway_mixed`.
- Grouped query noise is implemented for static workload groups with diagonal
  variance bookkeeping.

Next Recommended Task:

- Add a named experiment config for expanded high-order workloads, including a
  deliberate privacy allocation across `kway`, `kway_prefix`, `kway_range`,
  `kway_mixed`, and existing low-order families.
- Decide whether future adaptive measurement needs correlated group covariance
  bookkeeping or whether the current diagonal zCDP group mechanism is
  sufficient for the first paper experiments.

## Configurable K-Way Workloads - 2026-06-17

Task:

- Add configurable high-order workload families beyond the previous fixed
  two-way/mixed construction.

Implemented:

- Added `include_kway`.
  - Family: `kway`.
  - Generates sampled k-way equality conjunctions.
  - Controlled by `kway_orders` and `kway_queries_per_order`.
- Added `include_kway_prefix`.
  - Family: `kway_prefix`.
  - Generates sampled k-way conjunctions with one numeric `OP_LE` prefix term
    and equality terms on the remaining attributes.
  - Controlled by `kway_prefix_orders` and
    `kway_prefix_queries_per_order`.
- Added `include_kway_range`.
  - Family: `kway_range`.
  - Generates sampled k-way conjunctions with one numeric `OP_RANGE` term and
    equality terms on the remaining attributes.
  - Controlled by `kway_range_orders` and `kway_range_queries_per_order`.
- Extended `run_qdte` workload summaries and held-out workload config defaults
  so the new family controls are preserved in run artifacts.

Important Usage Note:

- These new families are disabled by default.
- In DP mode, enabling any of them requires adding matching
  `privacy.measurement_allocation` entries for `kway`, `kway_prefix`, and/or
  `kway_range`; otherwise measurement budget allocation correctly fails fast.
- The local-table consistency projection already supports these queries through
  the existing `QueryCatalogue` `EQ/LE/RANGE` representation, subject to
  `projection.consistency.max_scope_cells`.

Changed Files:

- `qdte/queries/workload.py`
- `qdte/evolution/engine.py`
- `tests/test_workload.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/queries/workload.py tests/test_workload.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_workload.py`
  - `3 passed in 0.04s`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_kway_workload_probe --workload.include_oneway false --workload.include_2way_cat false --workload.include_prefix false --workload.include_range false --workload.include_mixed false --workload.include_kway true --workload.include_kway_prefix true --workload.include_kway_range true --workload.kway_orders 3 --workload.kway_prefix_orders 3 --workload.kway_range_orders 4 --workload.kway_queries_per_order 12 --workload.kway_prefix_queries_per_order 12 --workload.kway_range_queries_per_order 12 --workload.max_queries 60 --privacy.measurement_allocation.kway 0.34 --privacy.measurement_allocation.kway_prefix 0.33 --privacy.measurement_allocation.kway_range 0.33 --qdte.max_iters 3 --qdte.stop_patience 3 --qdte.log_every 1 --evaluation.save_synthetic_csv false --runtime.xla_preallocate false`
  - passed
  - workload: `36` queries, with `12` each for `kway`, `kway_prefix`, and
    `kway_range`
  - initial measured loss: `542.61`
  - final measured loss: `370.4288896128588`
  - candidates scored: `768`
  - accepted edits: `24`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `158 passed in 10.33s`

Current Status:

- Workload generation now supports configurable high-order equality, prefix,
  and range conjunctions.
- Existing configs are unchanged because the new families default to disabled.

Next Recommended Task:

- Add a final experiment config that enables these families at a controlled
  order/query budget and allocates privacy budget across the expanded family
  set.

## Completed Algorithm Components Summary - 2026-06-17

Task:

- Summarize which algorithmic components of QDTE are currently completed.

Completed Algorithm Parts:

- Query/workload layer:
  - supports `oneway`, `twoway`, `prefix`, `range`, `mixed`, `kway`,
    `kway_prefix`, `kway_range`, `kway_mixed`,
    `orthogonal_kway_mixed`, and `halfspace`;
  - `QueryCatalogue` represents ordinary `EQ/LE/GE/RANGE` conjunctions and
    linear halfspace predicates.
- DP measurement layer:
  - static-all DP measurement with zCDP Gaussian noise;
  - per-family/group budget allocation and `rho_spent` reporting;
  - DP boundary keeps exact true answers evaluation-only.
- Projection layer:
  - simplex projection for partition groups;
  - clipping and prefix monotonicity projection;
  - query-space LSQ/feasible LSQ consistency projection;
  - local-table feasible consistency projection with CPU SLSQP and JAX dense
    active-set variants;
  - halfspace queries are included in local-table consistency via scope masks.
- Single-dataset QDTE generation layer:
  - objective invariants implemented:
    `residual = target_projected - answer_syn`,
    measured loss, candidate delta, and edit advantage;
  - active query selection using noisy/projected residuals and variances;
  - CPU directed enter/exit candidate repair;
  - CPU variants for masked, paired, exit-side, residual-weighted,
    enumerated-local, constructive partner, protected repair, and bounded
    best-partner diagnostics;
  - exact single-edit and aggregate edit-advantage scoring;
  - microbatch greedy, atom-flow, random group, directed group, and
    constructive-pair transports;
  - incremental residual update with recompute drift checks.
- Current practical generator:
  - A `constructive_pair_accept_anneal32` is the leading practical smoke
    generator under projected targets;
  - A can be explained as efficient pool-based/implicit partner construction.
- Interpretability/diagnostic generator:
  - D `bounded_best_partner` explicitly searches partner edits for harmed
    queries;
  - D is useful for mechanism explanation but remains slower than A.
- GPU/high-throughput layer:
  - fused `jax_repair` / `gpu_repair` candidate generation and scoring;
  - dense GPU, query-block GPU, and sparse-delta GPU scoring;
  - multi-word query-scope bitsets remove the old 31-attribute sparse GPU
    limitation;
  - GPU fused single-query repair now supports halfspace enter/exit.
- Evaluation/experiment support:
  - measured-loss and per-family metrics;
  - offline true-query and held-out evaluation;
  - candidate diagnostics;
  - ablation scripts and experiment-plan documents.

Not Yet Completed:

- Reusable select-measure-generate artifact boundary.
- Adaptive measurement/query selection with privacy accounting.
- Outer dataset-level population evolution wrapping inner QDTE/A.
- Exact upstream Private-GSD integration as a paper baseline.
- Downstream ML evaluation and public schema loader.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation summary only; no tests were run.

Current Status:

- The completed core is a single-dataset, residual-directed QDTE generator with
  consistency-projected DP targets and extensive ablation variants.
- The next algorithmic build should be the SMG boundary, then the outer
  dataset-population evolution layer.

Next Recommended Task:

- Refactor or wrap `run_qdte` into reusable select/measure/project/generate
  phases so multiple generators and future population individuals can consume
  the same measured target artifact.

## Halfspace Consistency Projection Clarification - 2026-06-17

Question:

- Whether halfspace queries are included in the consistency projection together
  with other query families.

Answer:

- Yes. Halfspace support was already wired into the measurement/projection
  layer before the GPU fused repair change.
- In `qdte/measurement/consistency.py`, `_query_scope` includes both ordinary
  query terms and `qcat.linear_terms(qid)`, so a halfspace query is assigned to
  the local table over its linear attributes.
- `_mask_indices` turns a halfspace predicate into the subset of local-table
  cells satisfying `sum_i weight_i * x[attr_i] <= threshold`.
- For `local_table_feasible_lsq` and `local_table_feasible_jax`, halfspace
  query masks are included in the same weighted measurement matrix as the other
  queries, and overlapping scopes are tied by shared marginal equality
  constraints.
- Therefore a halfspace such as `(a + x <= 2)` and one-way queries on `a` are
  reconciled through overlapping local scopes, subject to the configured
  `projection.consistency.max_scope_cells` limit.

Changed Files:

- `docs/HANDOFF.md`

Tests Run:

- Documentation clarification only; no tests were run for this answer.

Current Status:

- The recent GPU fused repair work changed candidate generation/scoring support
  for halfspace, not the consistency projection layer.
- The key caveat remains scope size: high-dimensional halfspace queries can make
  their local table too large and will fail fast if they exceed
  `projection.consistency.max_scope_cells`.

## GPU Fused Halfspace Repair - 2026-06-17

Task:

- Implement halfspace support in the high-throughput GPU fused candidate repair
  path.

Implemented:

- Added JAX halfspace scoring/repair helpers in
  `qdte/evolution/gpu_candidates.py`.
- `_eval_candidate_source_satisfaction` now evaluates both ordinary
  `EQ/LE/GE/RANGE` terms and halfspace linear predicates when filtering GPU
  source draws.
- `_repair_directed_rows` now applies directed halfspace enter/exit repair:
  - positive residuals reduce the halfspace linear score until
    `score <= threshold` when feasible;
  - negative residuals increase the score until `score > threshold` when
    feasible;
  - exit repair keeps the existing random-mutation fallback when a halfspace
    cannot be exited by directed linear-term changes.
- GPU dense and query-block scoring now pass `linear_attrs`,
  `linear_weights`, `linear_thresholds`, and `linear_num_terms` into
  `eval_records_queries_arrays`, so halfspace predicates are included in
  `dense_gpu` scoring as well as `sparse_delta_gpu`.
- Removed the config-validation fail-fast that previously rejected
  `workload.include_halfspace=true` with `candidate_backend=jax_repair` or
  `gpu_repair`.
- Added two small performance optimizations:
  - source filtering now skips the linear halfspace loop when the active batch
    has no halfspace queries;
  - halfspace-only active batches skip the ordinary `EQ/LE/GE/RANGE` repair
    loop, and halfspace repair reuses gathered linear query arrays across its
    greedy repair steps.

Scope:

- Supported now:
  - `candidate_backend: jax_repair` / `gpu_repair`;
  - default `candidate_compiler: single_query`;
  - `score_backend: dense_gpu` and `sparse_delta_gpu`;
  - halfspace-only queries and mixed catalogues containing halfspace queries.
- Still CPU-only:
  - structured compilers other than `single_query`, such as paired/masked/D
    variants, because config validation already requires those compilers to use
    `candidate_backend=cpu_repair`.

Changed Files:

- `qdte/evolution/gpu_candidates.py`
- `qdte/config_validation.py`
- `tests/test_gpu_candidates.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/gpu_candidates.py qdte/config_validation.py tests/test_gpu_candidates.py tests/test_config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_gpu_candidates.py tests/test_config_validation.py`
  - `65 passed in 4.78s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `156 passed in 10.44s`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_halfspace_gpu_repair_probe --workload.include_halfspace true --workload.halfspace_queries 8 --privacy.measurement_allocation.oneway 0.20 --privacy.measurement_allocation.twoway 0.20 --privacy.measurement_allocation.prefix 0.15 --privacy.measurement_allocation.range 0.15 --privacy.measurement_allocation.mixed 0.15 --privacy.measurement_allocation.halfspace 0.15 --qdte.candidate_backend jax_repair --qdte.score_backend sparse_delta_gpu --qdte.total_candidates_per_iter 256 --qdte.num_active_targets 8 --qdte.max_iters 5 --qdte.stop_patience 5 --qdte.log_every 1 --runtime.use_pmap true --runtime.xla_preallocate false`
  - passed
  - initial measured loss: `3458.04`
  - final measured loss: `2628.71`
  - candidates scored: `1280`
  - accepted edits: `40`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_halfspace_gpu_repair_probe_opt --workload.include_halfspace true --workload.halfspace_queries 8 --privacy.measurement_allocation.oneway 0.20 --privacy.measurement_allocation.twoway 0.20 --privacy.measurement_allocation.prefix 0.15 --privacy.measurement_allocation.range 0.15 --privacy.measurement_allocation.mixed 0.15 --privacy.measurement_allocation.halfspace 0.15 --qdte.candidate_backend jax_repair --qdte.score_backend sparse_delta_gpu --qdte.total_candidates_per_iter 256 --qdte.num_active_targets 8 --qdte.max_iters 5 --qdte.stop_patience 5 --qdte.log_every 5 --runtime.use_pmap true --runtime.xla_preallocate false`
  - passed
  - final measured loss: `2628.707164028321`
  - wall clock seconds: `7.051303271000506`
  - candidates scored: `1280`
  - accepted edits: `40`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/smoke_gpu_repair_ordinary_probe_opt --qdte.candidate_backend jax_repair --qdte.score_backend sparse_delta_gpu --qdte.total_candidates_per_iter 256 --qdte.num_active_targets 8 --qdte.max_iters 5 --qdte.stop_patience 5 --qdte.log_every 5 --runtime.use_pmap true --runtime.xla_preallocate false`
  - passed
  - final measured loss: `3254.315521274113`
  - wall clock seconds: `7.034338375000516`
  - candidates scored: `1280`
  - accepted edits: `40`

Current Status:

- GPU fused single-query candidate repair now supports halfspace workloads.
- The old documentation statements that halfspace requires CPU repair are
  superseded by this section.

Next Recommended Task:

- Run a short highpower workload with halfspace enabled and compare measured
  loss/time against the CPU repair backend.
- Use a longer warmup than the 5-step smoke probes before drawing performance
  conclusions, because the short probes are dominated by JAX compilation.

## GitHub Sync Preparation - 2026-06-17

Task:

- Sync the current workspace content to the remote GitHub repository.

Changed Files Included In This Sync:

- Documentation:
  - `README.md`
  - `architecture.md`
  - `architecture_zh.md`
  - `docs/HANDOFF.md`
  - `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
  - `docs/QDTE_ABLATION_SUMMARY.md`
  - `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- Configs:
  - `configs/adult_qdte.yaml`
  - `configs/adult_qdte_gpu_highpower.yaml`
- Core code:
  - `qdte/config_validation.py`
  - `qdte/eval/runtime.py`
  - `qdte/evolution/candidates.py`
  - `qdte/evolution/engine.py`
  - `qdte/evolution/gpu_candidates.py`
  - `qdte/evolution/scheduler.py`
  - `qdte/evolution/scoring.py`
  - `qdte/evolution/transport.py`
  - `qdte/measurement/consistency.py`
  - `qdte/measurement/measure.py`
  - `qdte/measurement/projection.py`
  - `qdte/queries/delta_index.py`
  - `qdte/queries/eval_jax.py`
  - `qdte/queries/types.py`
  - `qdte/queries/workload.py`
- Scripts:
  - `scripts/run_ablation.py`
  - `scripts/run_robustness_matrix.py`
  - `scripts/summarize_candidate_diagnostics.py`
- Tests:
  - `tests/test_config_validation.py`
  - `tests/test_consistency_projection.py`
  - `tests/test_delta_index.py`
  - `tests/test_engine_smoke.py`
  - `tests/test_gpu_candidates.py`
  - `tests/test_measurement.py`
  - `tests/test_projection.py`
  - `tests/test_queries.py`
  - `tests/test_repairs.py`
  - `tests/test_scheduler.py`
  - `tests/test_transport.py`
  - `tests/test_workload.py`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `153 passed in 8.35s`

Current Status:

- `outputs/` is ignored and is not part of this sync.
- The sync target is `origin/main`
  (`git@github.com:houyuwushang/private-de.git`).
- The workspace is ready to commit and push after a final diff check.

Next Recommended Task:

- After the remote sync, start implementing the select-measure-generate
  boundary:
  - reusable measured-target artifact save/load;
  - generator entry point that consumes an existing measured target;
  - later, the outer dataset-population wrapper around inner A/QDTE.

## Select-Measure-Generate And Outer Evolution Planning - 2026-06-17

Task:

- Decide whether generation-algorithm exploration is far enough to move on to
  the select-measure-generate framework.
- Clarify that the outer dataset-level evolution layer has not yet been added.

Conclusion:

- Treat the current generation exploration as provisionally complete.
- Use A, especially `constructive_pair_accept_anneal32`, as the first practical
  generator candidate.
- Keep D as the explicit/interpretable partner-construction ablation, not the
  first practical default, unless later multi-seed runs reverse the tradeoff.
- The repository currently has a single-dataset QDTE generator plus robustness
  scripts and Private-GSD-style internal baselines. It does not yet have a
  first-class two-level algorithm where an outer population of synthetic
  datasets wraps the inner QDTE generator.

Updated Plan:

- Added `Select-Measure-Generate Framework Roadmap` to
  `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`.
- Added `Outer Dataset-Level Evolution` to
  `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`.
- The framework boundary should be:
  - select workload/query groups;
  - measure/project once under the DP budget;
  - generate one or more synthetic datasets as post-processing of the same
    noisy/projected target.
- The outer evolution boundary should be:
  - maintain a population of synthetic dataset states;
  - share one fixed measured target across all individuals;
  - run short inner QDTE phases per individual;
  - select/restart/crossover individuals using measured loss only;
  - compare against inner-only A by equal candidate evaluations and equal wall
    time.

Changed Files:

- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- `docs/HANDOFF.md`

Tests Run:

- Documentation update only; code tests were not run.

Current Status:

- The immediate next engineering step is to refactor or wrap `run_qdte` into
  reusable phases so a fixed measured target can be reused by multiple
  generators and later by a population wrapper.
- The DP boundary remains the key invariant: the outer dataset population must
  not remeasure real data and must not use offline true-query metrics for
  active selection, restart, stopping, or hyperparameter choice.

Next Recommended Task:

- Implement the SMG boundary first:
  - measured-target artifact save/load;
  - generator entry point that consumes an existing measured target;
  - commands that run several generator variants against exactly the same
    target.
- Then add the outer population wrapper:
  - `inner_only_A`;
  - random-restart ensemble;
  - population A with elite selection and restarts;
  - population random-mutation baseline.

## A Current Consistency Rerun And D-Style Narrative - 2026-06-17

Task:

- Recheck the earlier A results under the same `local_table_feasible_jax`
  projected target used for D.
- Answer whether A is much faster than D and whether A can be explained with
  the D-style partner-construction narrative.

Runs:

- A fixed-8:
  - Command variant: `constructive_pair`
  - Output: `outputs/exp_a_current_consistency_2000_constructive_pair/`
  - Projection: `projection.consistency.enabled=true`,
    `projection.consistency.method=local_table_feasible_jax`
  - Final measured loss: `2.2138641508483943`
  - True RMSE: `0.003275422882885261`
  - True MAE: `0.0022674897119341563`
  - Accepted edits: `1535`
  - Scored candidates: `512000`
  - Generation seconds: `15.750771099003032`
  - Wall clock seconds: `49.76326872699428`
- A anneal32:
  - Command variant: `constructive_pair_accept_anneal32`
  - Output:
    `outputs/exp_a_current_consistency_2000_constructive_pair_accept_anneal32/`
  - Projection: `projection.consistency.enabled=true`,
    `projection.consistency.method=local_table_feasible_jax`
  - Final measured loss: `1.932742978022503`
  - True RMSE: `0.0032954638982284038`
  - True MAE: `0.002242798353909465`
  - Accepted edits: `1750`
  - Scored candidates: `512000`
  - Generation seconds: `16.162006594997365`
  - Wall clock seconds: `50.39548583002761`

Interpretation:

- A is much faster than D on the same projected smoke target:
  - A generation time is about `16s`;
  - dense D generation time is `116.45s`;
  - cached D generation time is `82.82s`;
  - JAX-batch D generation time is `73.98s`.
- A anneal32 has the best measured loss in the current projected smoke table:
  `1.932743`, compared with dense D's `1.997482`.
- Dense D has slightly better offline true-query RMSE/MAE than A anneal32:
  `0.003235`/`0.002218` vs `0.003295`/`0.002243`.
- A can be narrated using D's logic, but the distinction should be explicit:
  - D explicitly searches/synthesizes a partner edit for each seed edit to
    repair harmed queries.
  - A is the efficient pool-based version: it creates a residual-directed edit
    pool and selects pair/prefix bundles using exact aggregate edit advantage.
  - A therefore implements implicit/opportunistic partner construction rather
    than per-seed custom partner synthesis.

Changed Files:

- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- Experiment reruns only; no code tests were required for this documentation
  update.

Current Status:

- Current code reproduces the earlier A results under
  `local_table_feasible_jax`.
- A remains the strongest practical smoke result by speed and measured loss.
- D remains the clearest explanation/mechanism ablation for explicit
  record-level partner construction.

Next Recommended Task:

- For final evidence, run a frozen multi-seed projected-target matrix covering:
  - `random_best_edit`;
  - A fixed-8;
  - A anneal32;
  - dense D reference;
  - cached or JAX-batch D as performance variants.
- In the paper framing, use "individual-level directed edit-bundle
  construction" as the shared principle, with A as the practical algorithm and
  D as the explicit partner-construction ablation unless later multi-seed
  results reverse the tradeoff.

## D Precision-Preserving Performance Probe - 2026-06-17

Task:

- Continue optimizing D `bounded_best_partner` while checking whether speedups
  preserve dense-D quality/trajectory.
- The user emphasized that D has the clearest explanation for the paper:
  record-level directed construction of a partner edit, so performance should
  be treated as an engineering bottleneck rather than as a reason to drop D.

Implemented:

- Added additional D backends:
  - `best_partner_delta_backend: dense_cached`
    - caches `phi(X_syn)` for current source rows and reuses old-row query
      values when scoring D partner candidates;
    - keeps new-row query evaluation on the existing NumPy path.
  - `best_partner_delta_backend: source_cached`
    - only caches source-filter query satisfaction;
    - leaves delta and pair scoring exactly on dense CPU.
  - `best_partner_delta_backend: dense_unique`
    - computes dense CPU deltas once per unique `(old_row, new_row)` pair and
      maps them back to original order.
  - `best_partner_delta_backend: jax_fixed`
    - keeps per-seed D search but uses fixed-shape JAX delta evaluation.
- Added explicit ablation variants:
  - `bounded_best_partner_cached`
  - `bounded_best_partner_cached_verified`
  - `bounded_best_partner_source_cached`
  - `bounded_best_partner_unique`
  - `bounded_best_partner_gpu_exact`
  - `bounded_best_partner_jax_fixed`
- Added `best_partner_cached_verify_margin`.
  - Default is `0.0`.
  - Nonzero values turn cached D into a verified/top-rescore variant, so it is
    not a strict dense-D replacement.

Runs:

- Dense D 100-step reference:
  - Output: `outputs/exp_d_backend_probe_dense_bounded_best_partner/`
  - final measured loss: `19.142528412208623`
  - candidate generation seconds: `21.596446293930057`
  - generation seconds: `41.137639205990126`
- `dense_cached` 100-step:
  - Output: `outputs/exp_d_backend_probe_cached_bounded_best_partner_cached/`
  - exact same final metrics as dense D 100-step
  - candidate generation seconds: `12.674661150027532`
  - generation seconds: `32.619206096976995`
- `dense_cached` 2000-step:
  - Output: `outputs/exp_cd_consistency_2000_cached_bounded_best_partner_cached/`
  - final measured loss: `2.021054718885523`
  - true RMSE: `0.0032942149067496872`
  - true MAE: `0.0022839506172839504`
  - candidate generation seconds: `48.38959626774886`
  - generation seconds: `82.82311768399086`
  - wall clock seconds: `117.23676445402089`
- Dense D 2000-step reference:
  - Output: `outputs/exp_cd_consistency_2000_bounded_best_partner/`
  - final measured loss: `1.9974821332052242`
  - true RMSE: `0.0032349684041937236`
  - true MAE: `0.002218106995884773`
  - candidate generation seconds: `82.37158754112897`
  - generation seconds: `116.44865611300338`
  - wall clock seconds: `150.75503508999827`
- `source_cached` 100-step:
  - Output: `outputs/exp_d_backend_probe_source_cached_bounded_best_partner_source_cached/`
  - exact same final metrics as dense D 100-step
  - candidate generation seconds: `21.786897285928717`
  - no speedup on smoke
- `dense_unique` 100-step:
  - Output: `outputs/exp_d_backend_probe_unique_bounded_best_partner_unique/`
  - exact same final metrics as dense D 100-step
  - candidate generation seconds: `22.548881267168326`
  - slower than dense on smoke
- `jax_fixed` 100-step:
  - Output: `outputs/exp_d_backend_probe_jaxfixed_bounded_best_partner_gpu_exact/`
  - final measured loss: `18.452575913169508`
  - candidate generation seconds: `13.159871109004598`
  - faster but not bitwise-equivalent
- `dense_cached` with `best_partner_cached_verify_margin=1e-4`, 100-step:
  - Output: `outputs/exp_d_backend_probe_cached_verify_bounded_best_partner_cached/`
  - final measured loss: `17.724721913360675`
  - candidate generation seconds: `24.433369751961436`
  - changed the optimization path; keep as a separate variant only

Interpretation:

- There is a real engineering bottleneck in D partner scoring:
  - dense D 2000-step candidate generation: `82.37s`
  - cached D 2000-step candidate generation: `48.39s`
  - JAX-batch fast D 2000-step candidate generation from the previous section:
    `27.27s`
- Strict bitwise-equivalent speedups were not successful on the smoke setup:
  - `source_cached` was exact but did not speed up;
  - `dense_unique` was exact but slower.
- `dense_cached` is the most useful conservative speed/quality tradeoff:
  - same 100-step trajectory as dense D;
  - 2000-step quality remains close (`2.02105` vs `1.99748`) but not exactly
    identical because tiny numerical/tie differences compound in the late
    stochastic search.
- For paper claims:
  - keep dense D as the quality/reference definition;
  - report cached/GPU D as implementation accelerations or performance
    variants, unless a later multi-seed sweep shows cached D is statistically
    indistinguishable.

Changed Files:

- `qdte/evolution/candidates.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/config_validation.py scripts/run_ablation.py tests/test_repairs.py tests/test_config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py::test_bounded_best_partner_scores_and_attaches_best_partner tests/test_config_validation.py::test_bounded_best_partner_controls_are_allowed`
  - `6 passed`

Current Status:

- D now has a clear reference implementation and several speed variants.
- The best current practical speed variant is `bounded_best_partner_cached`.
- The best current strict-quality reference remains `bounded_best_partner`.

Next Recommended Task:

- For a real decision, run multi-seed comparisons of:
  - dense D reference;
  - cached D;
  - GPU-batch D;
  - A fixed-8 and A anneal32.
- If cached D remains within noise of dense D across seeds, use cached D for
  larger experiments and describe it as a memoized implementation of D.

## D GPU Batch Optimization - 2026-06-17

Task:

- Improve the efficiency of D `bounded_best_partner`, specifically the expensive
  partner candidate scoring path.

Implemented:

- Added `best_partner_delta_backend: jax_batch`.
  - This is a seed-chunked JAX/GPU path for D partner scoring.
  - It collects partner candidates for a chunk of seed edits, evaluates partner
    deltas in fixed-size JAX batches, then scores pair advantage against each
    seed's delta.
  - Fixed-size padding is used to avoid many XLA recompiles from variable
    partner batch shapes.
- Kept the old D path available:
  - `best_partner_delta_backend: dense_cpu`
  - `best_partner_delta_backend: sparse_cpu`
  - `best_partner_delta_backend: jax`
- Kept `scripts/run_ablation.py --variant bounded_best_partner` on the old
  dense D default so previous quality results remain reproducible.
- Added explicit fast variants:
  - `bounded_best_partner_gpu`
  - `best_partner_d_gpu`
  - `bounded_best_partner_fast`

Performance Runs:

- 100-step dense D probe:
  - Output: `outputs/exp_d_backend_probe_dense_bounded_best_partner/`
  - final measured loss: `19.142528412208623`
  - runtime generation seconds: `41.137639205990126`
  - candidate generation seconds: `21.596446293930057`
- 100-step JAX-batch D probe:
  - Output: `outputs/exp_d_backend_probe_jaxbatch_bounded_best_partner/`
  - final measured loss: `13.09147696570265`
  - runtime generation seconds: `31.88090797199402`
  - candidate generation seconds: `7.533141536900075`
- 2000-step dense D reference:
  - Output: `outputs/exp_cd_consistency_2000_bounded_best_partner/`
  - final measured loss: `1.9974821332052242`
  - true RMSE: `0.0032349684041937236`
  - runtime generation seconds: `116.44865611300338`
  - candidate generation seconds: `82.37158754112897`
  - wall clock seconds: `150.75503508999827`
- 2000-step JAX-batch fast D:
  - Output: `outputs/exp_cd_consistency_2000_jaxbatch_bounded_best_partner/`
  - final measured loss: `2.1536425253131948`
  - true RMSE: `0.0033048162883265505`
  - runtime generation seconds: `73.9818068909808`
  - candidate generation seconds: `27.26916706157499`
  - wall clock seconds: `108.09290260801208`

Interpretation:

- The GPU-batched path is a real speed improvement:
  - candidate generation: `82.37s -> 27.27s`
  - generation runtime: `116.45s -> 73.98s`
  - wall clock: `150.76s -> 108.09s`
- It is not a bitwise-equivalent optimization:
  - seed-chunking changes the partner-search stochastic path and final loss;
  - dense D remains the quality reference for the existing D result.
- The fast path is still useful for sweeps and larger workloads where dense
  partner scoring dominates runtime.

Changed Files:

- `qdte/evolution/candidates.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/config_validation.py scripts/run_ablation.py tests/test_repairs.py tests/test_config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py::test_bounded_best_partner_scores_and_attaches_best_partner tests/test_config_validation.py::test_bounded_best_partner_controls_are_allowed`
  - `3 passed`

Current Status:

- D has both a quality/reference dense implementation and an explicit faster
  GPU-batched implementation.
- `docs/QDTE_ABLATION_SUMMARY.md` now separates the two rows.

Next Recommended Task:

- If D remains important, tune `best_partner_seed_batch_size` and
  `best_partner_jax_batch_size` on the target workload; do not assume the smoke
  defaults are optimal for Adult/full workloads.

## C/D Consistency Projection Rerun - 2026-06-17

Task:

- Rerun the C and D routes after `local_table_feasible_jax` consistency
  projection, because previous C/D pilot losses used the no-consistency target
  and were not comparable to A/B.

Commands:

- C:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant protected_same_row --run.output_dir outputs/exp_cd_consistency_2000 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 500 --qdte.candidate_diagnostics true`
- D:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant bounded_best_partner --run.output_dir outputs/exp_cd_consistency_2000 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 500 --qdte.candidate_diagnostics true`

Outputs:

- C output:
  - `outputs/exp_cd_consistency_2000_protected_same_row/`
  - final measured loss: `11.509981871826184`
  - true RMSE: `0.004219784871547354`
  - true MAE: `0.002621399176954732`
  - accepted edits: `1140`
  - candidates scored: `512000`
  - runtime generation seconds: `73.00366302300245`
- D output:
  - `outputs/exp_cd_consistency_2000_bounded_best_partner/`
  - final measured loss: `1.9974821332052242`
  - true RMSE: `0.0032349684041937236`
  - true MAE: `0.002218106995884773`
  - accepted edits: `1169`
  - candidates scored: `520994`
  - runtime generation seconds: `116.44865611300338`

Interpretation:

- The previous no-consistency C/D table should not be used to claim C/D are
  worse than A/B; it optimized a different projected/noisy target.
- Under the fair `local_table_feasible_jax` target, C still stalls above A.
- D is now competitive: it beats fixed-8 A on measured loss on seed `0`
  (`1.997482` vs `2.213864`) and is close to annealed A (`1.932743`), but it is
  slower and uses side-budget partner scoring (`520994` scored candidates).
- This makes D a real candidate ablation rather than only a negative result.

Changed Files:

- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- Not run; this task only reran experiments and updated docs.

Current Status:

- Same-projection A/C/D/random-best comparison is now documented in
  `docs/QDTE_ABLATION_SUMMARY.md`.
- D should be included in the next multi-seed consistency-projection sweep.

Next Recommended Task:

- Run a frozen multi-seed sweep under one projection setting for:
  - `random_best_edit`
  - `constructive_pair`
  - `constructive_pair_accept_anneal32`
  - `bounded_best_partner`
  - optionally `protected_same_row` as the C diagnostic

## Random Best Loss Discrepancy - 2026-06-17

Task:

- Investigate why `random_best_edit` previously reached measured loss below
  `10`, but the later current-code table reported `66.624540`.

Finding:

- This was a projection-target mismatch, not an algorithm regression.
- The old `7.243474` `random_best_edit` run and the old A/B runs used:
  - `projection.consistency.enabled=true`
  - `projection.consistency.method=local_table_feasible_jax`
- The later `66.624540` current-code run used the default `configs/smoke.yaml`
  projection path without `projection.consistency`.
- Therefore the measured losses are against different projected/noisy targets
  and should not be ranked against each other.

Verification Run:

- Command:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant random_best_edit --run.output_dir outputs/exp_random_best_consistency_check_2000 --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 2000 --qdte.stop_patience 2000 --qdte.log_every 500 --qdte.candidate_diagnostics false`
- Output:
  - `outputs/exp_random_best_consistency_check_2000_random_best_edit/`
- Result:
  - final measured loss: `7.243473932646188`
  - true RMSE: `0.003832930735702298`
  - true MAE: `0.0025020576131687244`
  - accepted edits: `1175`
  - candidates scored: `512000`
- This exactly matches the old robustness-matrix `random_best_edit` / relabeled
  `private_gsd_mutate` result.

Changed Files:

- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- Not run; this was a config/result audit plus one reproduction experiment.

Current Status:

- `docs/QDTE_ABLATION_SUMMARY.md` now separates:
  - no-consistency current-code runs, where `random_best_edit=66.624540`;
  - `local_table_feasible_jax` runs, where `random_best_edit=7.243474`.
- Future A/B/C/D comparisons must freeze projection settings.
- Do not compare measured loss across different projection targets.

Next Recommended Task:

- For final method selection, rerun the candidate methods under one chosen
  projection setting:
  - if using the old A/B evidence, refresh `constructive_pair`,
    `constructive_pair_accept_anneal32`, `directed_group_advantage`,
    `protected_same_row`, and `bounded_best_partner` all with
    `local_table_feasible_jax`;
  - if using the no-consistency setting, rerun A and B under no consistency
    before claiming that A still beats current directed-group variants.

## C/D Pilot And Bounded Best-Partner D - 2026-06-16

Task:

- Continue the C and D route checks before launching any final full experiment matrix.
- Keep this as a pilot/smoke-level comparison, not a paper-ready multi-seed result.

Implemented:

- Added D-style `qdte.candidate_compiler: bounded_best_partner`.
  - Generates the same seed edit pool as A.
  - For each seed, computes harmed residual-weighted query support.
  - Samples partner sources on the opposite side of harmed queries.
  - Generates multiple partner repairs per source.
  - Scores full `delta_seed + delta_partner` using the QDTE objective.
  - Attaches only positive best partners as explicit pair units for the existing
    `constructive_pair` transport.
- Added `scripts/run_ablation.py --variant bounded_best_partner`.
  - Default D pilot uses `best_partner_delta_backend: dense_cpu`; the initial
    JAX backend attempt was stopped because variable-size bounded partner
    search did not enter the first metrics row after about 400 seconds.
- Added D diagnostics:
  - `best_partner_candidates`
  - `best_partner_pair_units`
  - `best_partner_seed_candidates`
  - `best_partner_source_attempts`
  - `best_partner_source_failures`
  - `best_partner_pairs_evaluated`
  - `best_partner_positive_pairs`
- Added repair type `18: bounded_best_partner`.
- Updated docs:
  - `docs/QDTE_ABLATION_SUMMARY.md`
  - `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`

Changed Files:

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- `docs/HANDOFF.md`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py tests/test_repairs.py tests/test_config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py::test_bounded_best_partner_scores_and_attaches_best_partner tests/test_config_validation.py::test_bounded_best_partner_controls_are_allowed`
  - `2 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_transport.py tests/test_config_validation.py`
  - `92 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `149 passed`

Runs:

- D 5-step smoke:
  - Output: `outputs/exp_bounded_best_partner_smoke_bounded_best_partner/`
  - Loss: `4306.66 -> 3242.21`
  - Accepted edits: `40`
- D dense-CPU 20-step smoke:
  - Output: `outputs/exp_bounded_best_partner_densecpu_smoke_bounded_best_partner/`
  - Loss: `4306.66 -> 1303.90`
  - Accepted edits: `160`
- C 2000-step pilot:
  - Output: `outputs/exp_cd_pilot_2000_protected_same_row/`
  - Final measured loss: `73.254116`
  - True RMSE: `0.004695`
  - True MAE: `0.002712`
  - Candidates scored: `512000`
  - Accepted edits: `1195`
  - Runtime generation seconds: `69.25`
- D 2000-step pilot:
  - Output: `outputs/exp_cd_pilot_2000_densecpu_bounded_best_partner/`
  - Final measured loss: `59.726185`
  - True RMSE: `0.003227`
  - True MAE: `0.002185`
  - Candidates scored: `522460`
  - Accepted edits: `1239`
  - Runtime generation seconds: `802.20`
  - Candidate generation seconds: `761.06`
  - D uses side-budget partner search, so it is not a strict 512000-candidate
    same-budget run.

Interpretation:

- C is not competitive in standalone form.
  - It still becomes too constrained late in optimization and stalls above the
    current random-best and directed-group pilots.
- D validates the best-partner idea directionally:
  - it is much better than C and better than current random best-single on this
    seed;
  - it is close to but slightly worse than `directed_group_advantage` max size
    `4` (`59.726185` vs `59.624552` measured loss);
  - it is far slower, with candidate generation dominating runtime.
- Current decision:
  - keep C and D as ablations/diagnostics;
  - do not replace A with D unless later full A refresh is unexpectedly weak or
    D can be made much cheaper.

Next Recommended Task:

- Refresh the historical A variants under the current code before making the
  final method choice:
  - `constructive_pair`
  - `constructive_pair_accept_anneal32`
- If A still lands near the historical `2.x` measured-loss range, use A as the
  main paper method and report C/D as negative or diagnostic ablations.

## Final Experiment Plan And C/D Route Scope - 2026-06-16

Task:

- Do not start long full-matrix experiments yet; current runs are still smoke or
  pilot evidence.
- Collect all experiments that should be run later into one planning document.
- Clarify how to continue evaluating C and D without weakening the current A
  route if A remains strongest.

Implemented:

- Added `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`.
  - Separates paper-level full runs from current smoke/historical results.
  - Defines freeze requirements, seed policy, candidate-budget policy, metrics,
    diagnostics, and reporting checklist.
  - Lists the main paper comparison, proposal-direction ablations, mask/paired
    ablations, edit-advantage blind controls, aggregate group-size sweeps, and
    projection/measurement ablations.
  - Gives a dedicated A/B2/C/D route plan:
    - A: current pool-level constructive pair, still the main candidate method;
    - B2: attached synthesized partner, keep only if it improves A under fair
      compute or reduces A's plateau;
    - C: protected same-row repair, useful if it becomes a cheap fallback or
      demonstrably reduces harmed-query collateral damage;
    - D: exact/bounded best partner, treat first as an oracle/diagnostic unless
      it beats A under fair budget.
- Updated `docs/QDTE_ABLATION_SUMMARY.md` to link to the final experiment plan.

Changed Files:

- `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`
- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`

Tests Run:

- Not run; documentation-only update.

Current Status:

- The experiment backlog is now documented separately from smoke results.
- The current decision rule is:
  - use A if it remains best or nearly best under full runs;
  - add A's annealed accept schedule if it improves consistently;
  - keep B2/C/D as ablations or diagnostics unless they beat A under fair
    candidate-evaluation and wall-time budgets.

Next Recommended Task:

- Continue C/D route analysis before launching full experiments:
  - inspect whether current `protected_same_row` is too constrained and whether
    a fallback-only C variant would be cleaner;
  - decide whether D should be a small-scope oracle, bounded beam search, or
    constraint-solver diagnostic;
  - only implement a new C/D variant if it tests a specific failure mode of A.

## Directed Group Advantage And Ablation Summary - 2026-06-16

Task:

- Summarize all intermediate QDTE ablations into a dedicated document for later full experiment planning.
- Implement a residual/delta-directed group-advantage transport, with group size as a tunable parameter.

Implemented:

- Added `docs/QDTE_ABLATION_SUMMARY.md`.
  - Separates current-code reruns from historical exploratory results.
  - Lists random baselines, directed single-edit variants, aggregate-advantage variants, constructive partner/repair variants, and a recommended final matrix.
- Added `transport_mode: directed_group`.
- Added `scripts/run_ablation.py --variant directed_group_advantage`.
- Directed group transport:
  - candidate edits can still be random;
  - seed edits are ranked by individual advantage;
  - each group is expanded greedily using virtual residual `residual - Delta_partial`;
  - marginal expansion uses the same QDTE objective:
    - `delta @ ((residual - Delta_partial) * inv_variance) - 0.5 * ((delta * delta) @ inv_variance) - lambda_cost * edit_cost`;
  - final acceptance uses exact aggregate group advantage;
  - row capacity is enforced inside each group.
- Added tunable config:
  - `directed_group_seed_count`
  - `directed_group_min_size`
  - `directed_group_max_size`
  - `directed_group_pool_multiplier`
  - `directed_group_max_pool`
  - `directed_group_allow_negative_steps`
- Added diagnostics:
  - `directed_group_pool_candidates`
  - `directed_group_seed_candidates`
  - `directed_group_groups_evaluated`
  - `directed_group_positive_groups`
  - `directed_group_expansion_steps`
  - `directed_group_best_group_size`
  - `directed_group_accepted_candidates`
  - `directed_group_groups_with_negative_member`

Changed Files:

- `docs/QDTE_ABLATION_SUMMARY.md`
- `docs/HANDOFF.md`
- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_transport.py`
- `tests/test_config_validation.py`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/transport.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py tests/test_transport.py tests/test_config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py::test_directed_group_transport_builds_compensating_group tests/test_config_validation.py::test_directed_group_transport_controls_are_allowed`
  - `2 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py::test_directed_group_transport_builds_compensating_group tests/test_transport.py::test_random_group_transport_scores_aggregate_group_advantage tests/test_config_validation.py::test_directed_group_transport_controls_are_allowed`
  - `3 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py tests/test_config_validation.py`
  - `62 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `145 passed`

Smoke:

- 5-step directed group smoke:
  - Output: `outputs/exp_directed_group_smoke_directed_group_advantage/`
  - Loss: `4306.66 -> 3338.32`
  - Accepted edits: `40`
  - Compared with the earlier random-group smoke loss `3965.07`, directed group gives a much faster early drop.

2000-step current-code sweep:

All runs use `configs/smoke.yaml`, seed `0`, `privacy.mode=dp`, 512000 scored candidates, random individual edit proposals, exact aggregate QDTE scoring, and offline true-query evaluation.

| Variant | Final measured loss | True RMSE | True MAE | Accepted edits | Wall seconds | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current `random_best_edit` | 66.624540 | 0.004038 | 0.002523 | 1150 | 4.61 | `outputs/exp_random_best_current_2000_random_best_edit/` |
| `random_group_advantage`, size `2..8` | 82.419445 | 0.005269 | 0.002918 | 2842 | 8.66 | `outputs/exp_random_group_2000_random_group_advantage/` |
| `random_group_advantage`, size `1..8` | 68.857956 | 0.004230 | 0.002634 | 2180 | 18.21 | `outputs/exp_random_group_min1_2000_random_group_advantage/` |
| `directed_group_advantage`, max size `2` | 59.628688 | 0.003222 | 0.002202 | 1655 | 11.12 | `outputs/exp_directed_group_g2_2000_directed_group_advantage/` |
| `directed_group_advantage`, max size `4` | 59.624552 | 0.003205 | 0.002148 | 1843 | 13.72 | `outputs/exp_directed_group_g4_2000_directed_group_advantage/` |
| `directed_group_advantage`, max size `8` | 59.627354 | 0.003232 | 0.002202 | 2035 | 14.47 | `outputs/exp_directed_group_g8_2000_directed_group_advantage/` |

Interpretation:

- The user's hypothesis is supported:
  - group-level aggregate advantage is useful when the group is constructed with residual/delta direction;
  - random group construction alone is not enough.
- On the current smoke setup, directed group improves over current random best-single:
  - measured loss: `66.624540 -> 59.624552`;
  - true RMSE: `0.004038 -> 0.003205`.
- Max group sizes `2`, `4`, and `8` are close.
  - Max size `4` is best in this sweep by measured loss and true RMSE.
  - The best accepted group is often smaller than the allowed max, so `directed_group_max_size` should remain a sweep parameter rather than a fixed claim.
- After these runs, the diagnostic semantics for `directed_group_positive_groups` were cleaned up to count seed-level best groups rather than positive prefixes.
  - Final loss/RMSE numbers above are unaffected.
  - If the paper needs detailed `directed_group_positive_groups` analysis, rerun the directed group sweep under the updated diagnostics.

Current Status:

- `random_group_advantage` and `directed_group_advantage` are both implemented and tested.
- The ablation summary doc is in place for planning final experiments.
- The most promising current-code result in this branch is `directed_group_advantage` with max group size `4`.

Next Recommended Task:

- Refresh current-code comparisons for historical strong methods under the same settings:
  - `constructive_pair`
  - `constructive_pair_accept_anneal32`
  - `constructive_partner_b2`
  - `protected_same_row`
- Then run a multi-seed matrix including:
  - `random_best_edit`
  - `random_group_advantage`
  - `directed_group_advantage` with max sizes `2`, `4`, `8`
  - the refreshed constructive methods.

## Random Group Advantage Probe - 2026-06-16

Task:

- Explore the user's suggestion: keep individual edit proposals random, but score randomly sampled edit groups with exact aggregate QDTE advantage.
- This is an exploratory probe, not a claim that random group search is the final QDTE innovation.

Implementation:

- Added `transport_mode: random_group`.
- Added `scripts/run_ablation.py --variant random_group_advantage`.
- Random group transport:
  - uses all finite candidate edits as the pool unless capped;
  - does not filter by positive individual advantage;
  - samples non-conflicting random groups;
  - scores each group by exact aggregate advantage:
    - `Adv(S) = Delta_S @ (residual * inv_variance) - 0.5 * ((Delta_S * Delta_S) @ inv_variance) - lambda_cost * cost_S`;
  - accepts only the best positive group.
- DP boundary is preserved:
  - no true answers are used for generation, scoring, group selection, stopping, or hyperparameter selection;
  - true-query metrics are offline evaluation only.

Changed Files:

- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_transport.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

New Diagnostics:

- `random_group_pool_candidates`
- `random_group_groups_evaluated`
- `random_group_positive_groups`
- `random_group_best_group_size`
- `random_group_accepted_candidates`
- `random_group_groups_with_negative_member`

Tests Run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/transport.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py tests/test_transport.py tests/test_config_validation.py`
  - passed
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py::test_random_group_transport_scores_aggregate_group_advantage tests/test_config_validation.py::test_random_group_transport_controls_are_allowed`
  - `2 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py tests/test_config_validation.py`
  - `60 passed`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - `143 passed`

Runs:

All runs used current code, `configs/smoke.yaml`, seed `0`, 2000 iterations, 512000 scored candidates, and offline true-query evaluation.

| Variant | Final measured loss | True RMSE | True MAE | Accepted edits | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_group_advantage`, size `2..8`, 128 groups/iter | 82.419445 | 0.005269 | 0.002918 | 2842 | 8.66 |
| `random_group_advantage`, size `1..8`, 512 groups/iter | 68.857956 | 0.004230 | 0.002634 | 2180 | 18.21 |
| `random_group_advantage`, size `1`, 512 groups/iter | 67.231262 | 0.004130 | 0.002494 | 1160 | 16.31 |
| current `random_best_edit` | 66.624540 | 0.004038 | 0.002523 | 1150 | 4.61 |

Interpretation:

- Pure random group scoring did not beat current random best-single edit on this setup.
- Allowing size `1` improves random group search, but it mostly collapses toward best-single behavior.
- Forcing group size at least `2` is substantially worse, even though it accepts more edits.
- Late in optimization, random groups almost never have positive aggregate advantage:
  - the final logged iteration for all group variants had `last_random_group_positive_groups = 0`.
- The old handoff number `random_best_edit = 7.243474` should not be used as the current-code same-condition reference.
  - Re-running `random_best_edit` now gives `66.624540` under the current code/config path.

Current Status:

- Random group advantage is implemented and tested as an ablation.
- The experiment supports the stronger claim that aggregate advantage alone is not enough.
- Useful groups need to be directionally constructed or searched with residual/delta structure; random grouping is too low-hit-rate.

Next Recommended Task:

- Implement a directed bundle search rather than more random grouping:
  - seed from high or near-high individual advantage edits;
  - expand with partners chosen to compensate the seed's harmed query deltas;
  - score the final bundle with the same exact aggregate advantage;
  - keep a random-reserve ablation only for comparison.

## Edit-Advantage Locality Analysis - 2026-06-16

Task:

- Assess whether the current edit-advantage gate may be too myopic and cause QDTE to slide into local optima.
- Clarify whether future work should focus on many separate A/B/C/D heuristics or on improving the edit-advantage search horizon.

Key code facts:

- `qdte/evolution/scoring.py` implements the required single-edit invariant:
  - `adv(e) = delta_e @ (residual * inv_variance) - 0.5 * ((delta_e * delta_e) @ inv_variance) - lambda_cost * edit_cost`.
- `qdte/evolution/transport.py` also implements exact bundle/batch advantage:
  - `Adv(S) = Delta_S @ (residual * inv_variance) - 0.5 * ((Delta_S * Delta_S) @ inv_variance) - lambda_cost * cost_S`.
- `constructive_pair` already uses aggregate pair and prefix-batch scoring, so the codebase has the right objective machinery for multi-edit scoring.

Current interpretation:

- The formula itself is not the likely bug.
  - For the current residual and a fixed candidate delta, the single-edit advantage is the exact measured-loss decrease under the QDTE quadratic objective.
- The real limitation is the search horizon.
  - Ranking and accepting isolated edits can reject moves that are bad alone but good when combined with compensating edits.
  - A bundle can have positive aggregate advantage even when one or more component edits have non-positive individual advantage, because collateral damage can cancel in `Delta_S`.
- This explains why A-style `constructive_pair` is much stronger than strict single-edit acceptance: it changes the scored object from one edit to a coordinated edit unit.
- C `protected_same_row` does not contradict this. It generated protected candidates, but late-stage candidates were too constrained and rarely had positive full QDTE advantage.

Recommended algorithm direction:

- Do not change the edit-advantage objective formula.
- Treat "better edit advantage search" as the main algorithmic line, rather than accumulating many unrelated A/B/C/D strategies.
- Avoid degenerating into Private-GSD-style random dataset search:
  - If QDTE randomly samples whole edit groups/datasets and only keeps the best measured loss, the distinction from GSD becomes weak.
  - The intended distinction is that QDTE uses `residual`, `inv_variance`, and per-edit `delta` structure to construct or expand a group directionally before scoring it.
  - Dataset-level scoring is acceptable only as a final exact check of a directionally constructed bundle, not as the primary source of exploration.
- Improve the proposal/transport horizon:
  - D1 bounded exact partner selection: for each promising or near-promising seed edit, generate/search a bounded partner pool and score `delta_seed + delta_partner` exactly.
  - Allow "negative seed only inside a bundle": a seed with small negative single-edit advantage may be kept for partner construction, but only accepted if the final aggregate bundle advantage is positive.
  - Beam or rollout transport for `k=3..K`: keep the top `B` partial bundles and score marginal gains against virtual residual `residual - Delta_partial`.
  - Keep the DP boundary: all scoring, partner selection, beam expansion, stopping, and hyperparameter choices must use only noisy/projected measurements, residuals, variances, deltas, and costs.
- Reframe A/B/C/D:
  - A is useful evidence that aggregate/pair advantage beats isolated edits.
  - B/C/D should be treated as candidate mechanisms for finding better aggregate deltas, not as independent algorithm families.
  - If a strategy cannot improve aggregate advantage search under a fixed budget, it should be dropped.

Changed Files:

- `docs/HANDOFF.md`

Tests / Commands Run:

- Read `qdte/evolution/scoring.py`, `qdte/evolution/transport.py`, and recent `docs/HANDOFF.md` sections.
- No tests run; this was a design/diagnosis update only.

Current Status:

- The edit-advantage invariant remains trusted.
- The likely improvement point is multi-edit lookahead/search, not replacing the objective.
- Main research framing should be "directed evolution by aggregate edit-advantage lookahead"; A/B/C/D are implementation variants of that idea.
- The key boundary against Private-GSD should be stated as:
  - Private-GSD-style baseline: random mutation/search at the dataset or population level, then global statistic error selection.
  - QDTE: residual-guided local construction of edit/bundle directions inside one synthetic dataset, then exact aggregate advantage scoring.

Next Recommended Task:

- Implement D1 as a bounded exact partner-selection transport/generator experiment, with diagnostics for:
  - negative or weak seeds considered;
  - partner pairs evaluated;
  - positive aggregate pairs;
  - accepted aggregate bundles;
  - final measured loss and offline true-query metrics.

## Protected Same-Row Repair C Failure Analysis - 2026-06-16

Task:

- Analyze why standalone `protected_same_row` C performs worse than A/B2 and even worse than `random_best_edit` on seed 0.

Key evidence:

- `protected_same_row` C 2000-step output:
  - `outputs/exp_protected_same_row_2000_protected_same_row/`
  - final measured loss: `11.509982`
  - true RMSE: `0.004220`
  - accepted edits: `1140`
  - wall time: `106.20s`
- Comparison:
  - A fixed-8 `constructive_pair`: loss `2.213864`, true RMSE `0.003275`, wall `49.08s`
  - A anneal 32->2: loss `1.932743`, true RMSE `0.003295`, wall `49.15s`
  - B2: loss `2.479907`, true RMSE `0.003322`, wall `118.79s`
  - `random_best_edit`: loss `7.243474`, true RMSE `0.003833`, wall `36.97s`
- C timeseries:
  - iter 100: loss `45.902027`, positive rate `0.074219`, accepted `5`, protected candidates `116`
  - iter 500: loss `12.256074`, positive rate `0.0`, accepted `0`, protected candidates `113`
  - iter 1000: loss `11.962106`, positive rate `0.0`, accepted `0`, protected candidates `121`
  - iter 2000: loss `11.509982`, positive rate `0.0078125`, accepted `2`, protected candidates `121`
- Logged-window averages after iter 100:
  - positive returned rate is nearly zero;
  - accepted edits are almost always zero;
  - protected candidates continue to consume roughly half the candidate pool.
- Family losses show the main failure is mixed queries:
  - C mixed loss: `8.9229`
  - A fixed-8 mixed loss: `0.9842`
  - A anneal 32->2 mixed loss: `0.8094`
  - B2 mixed loss: `1.2773`

Interpretation:

- C is not failing because it cannot generate protected candidates.
  - It keeps generating around `110-120` protected candidates per logged iteration.
- C fails because the protected same-row candidate space becomes too narrow late in optimization.
  - The protection rule tries to keep `phi_q(x_new) == phi_q(x_old)` for harmed queries.
  - In overlapping workloads, especially mixed/twoway queries, this often conflicts with useful target movement.
  - The candidate may preserve a harmed query, but it also removes degrees of freedom needed to repair other residuals.
- C also consumes candidate budget.
  - The standalone variant uses roughly half the directed budget for protected candidates.
  - After iter 100 these candidates rarely have positive full QDTE advantage, so they starve the normal directed/pair pool.
- Same-row protection is structurally weaker than pair repair.
  - A/B2 can use two records:
    - one record moves toward the target;
    - another record compensates collateral damage.
  - C forces one record to do both, which is often infeasible or too restrictive.
- The greedy protection order is not an objective optimizer.
  - It protects top harmed memberships locally.
  - It does not solve the best tradeoff between target gain and collateral loss.
  - Full edit advantage is still checked later, and most late protected candidates fail that check.
- Runtime is also worse.
  - C performs per-seed delta analysis plus several target/protection repair passes.
  - This makes it about `2.16x` slower than A on the seed-0 2000-step smoke run for the same scored-candidate count.

What is probably not the explanation:

- This does not look like a DP-boundary issue.
  - C uses only residuals against noisy/projected measurements and inverse variances.
  - Exact true answers are only used for offline evaluation.
- This does not look like "no protected candidates were generated."
  - The problem is low late-stage useful advantage, not generation failure.

Recommended next step:

- Do not continue standalone C as the main path.
- Use C only selectively:
  - as a side-budget fallback when B2/D1 cannot find a partner source;
  - or only keep a protected variant if its full QDTE advantage is better than the seed's advantage.
- Implement D1 next:
  - generate a bounded partner pool per seed;
  - score `delta_seed + delta_partner` exactly before attach/keep;
  - keep only top positive aggregate partners.
- A possible future C improvement:
  - make protection soft instead of hard;
  - keep protected candidates in a side budget, not half of the main directed pool;
  - use exact edit advantage at generation time to keep protected variants only when they improve over the unprotected seed.

Changed files:

- `docs/HANDOFF.md`

Tests / commands run:

- Parsed `outputs/exp_protected_same_row_2000_protected_same_row/metrics_timeseries.csv`.
- Parsed `candidate_diagnostics_timeseries.csv` from the 5-step C smoke run.
- Compared `metrics_final.json`, `runtime.json`, and `metrics_by_family.json` for:
  - A fixed-8 `constructive_pair`
  - A anneal 32->2
  - B2 `constructive_partner_b2`
  - C `protected_same_row`
  - `random_best_edit`
- No code tests were run; analysis/documentation only.

Current status:

- C is implemented and validated mechanically, but standalone C is not algorithmically competitive.
- The next implementation target remains D1 bounded exact partner selection.

## Protected Same-Row Repair C Implementation - 2026-06-16

Task:

- Start implementing the C/D plan.
- Implement C first: protected same-row repair.

Implemented C:

- New candidate compiler:
  - `qdte.candidate_compiler: protected_same_row`
  - aliases:
    - `protected_repair`
    - `constructive_protected`
- New ablation variant:
  - `scripts/run_ablation.py --variant protected_same_row`
- New repair type:
  - `17: protected_same_row`
- Default integration for the variant:
  - candidate compiler: `protected_same_row`
  - transport: `constructive_pair`
  - prefix strategy: `best_advantage`

Algorithm:

- Generate normal `single_query` seed edits.
- For each seed edit:
  - compute `delta_seed[q] = phi_q(x_seed) - phi_q(x_old)`;
  - compute `weight[q] = residual[q] * inv_variance[q]`;
  - identify harmed queries where `delta_seed[q] * weight[q] < 0`;
  - take the top harmed queries by weighted damage.
- For the same row, run bounded random restarts:
  - repair the target query in the intended enter/exit direction;
  - repair harmed queries back toward the old row's membership;
  - require the target direction to still hold;
  - require at least one harmed-query delta to be eliminated.
- Final acceptance still uses the unchanged full QDTE edit advantage. No exact true answers are used for generation/scoring.

New config:

- `protected_repair_seed_fraction`
- `protected_repair_harm_queries`
- `protected_repair_restarts_per_seed`
- `protected_repair_max_protection_passes`
- `protected_repair_require_target_direction`
- `protected_repair_delta_backend`: `jax`, `dense_cpu`, or `sparse_cpu`

Diagnostics:

- `protected_same_row_candidates`
- `protected_repair_seed_candidates`
- `protected_repair_attempts`
- `protected_repair_target_failures`
- `protected_repair_protection_successes`
- Candidate diagnostics also expose `rtype_protected_same_row_*`.

Validation:

- Unit test proves the key C behavior:
  - a target-enter seed would also enter an overfit harmed query;
  - protected repair keeps the target-enter direction;
  - protected repair preserves the harmed query membership from the old row.
- 5-step smoke confirms full QDTE loop integration and diagnostics:
  - Output: `outputs/exp_protected_same_row_smoke_protected_same_row/`
  - Final measured loss: `3233.672601`
  - Accepted edits: `40`
  - First iteration:
    - protected candidates: `76`
    - protected positive candidates: `71`

Seed-0 2000-step smoke comparison:

All rows use `configs/smoke.yaml`, seed `0`, `local_table_feasible_jax`, 2000 iterations.

| Method | Final measured loss | True RMSE | True MAE | Accepted edits | Scored candidates | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `protected_same_row` C | 11.509982 | 0.004220 | 0.002621 | 1140 | 512000 | 106.20 |
| A fixed-8 `constructive_pair` | 2.213864 | 0.003275 | 0.002267 | 1535 | 512000 | 49.08 |
| A anneal 32->2 | 1.932743 | 0.003295 | 0.002243 | 1750 | 512000 | 49.15 |
| B2 `constructive_partner_b2` | 2.479907 | 0.003322 | 0.002267 | 1470 | 542388 | 118.79 |
| `random_best_edit` | 7.243474 | 0.003833 | 0.002502 | 1175 | 512000 | 36.97 |

Milestone losses:

| Method | iter<=100 | iter<=500 | iter<=1000 | iter<=2000 |
| --- | ---: | ---: | ---: | ---: |
| `protected_same_row` C | 45.902027 | 12.256074 | 11.962106 | 11.509982 |
| A fixed-8 `constructive_pair` | 39.892043 | 5.923445 | 2.626935 | 2.213864 |
| A anneal 32->2 | 7.942800 | 2.233513 | 2.044579 | 1.932743 |
| B2 `constructive_partner_b2` | 34.065774 | 3.373532 | 2.784843 | 2.479907 |
| `random_best_edit` | 2078.194516 | 180.407902 | 17.657672 | 7.243474 |

Interpretation:

- The C mechanism is implemented and works locally.
- C as a standalone candidate compiler is not competitive in this seed-0 smoke run.
- It is slower than A because each seed repair performs harmed-query delta analysis and multiple protection attempts.
- It likely over-constrains the same-row search:
  - many candidates preserve harmed queries;
  - but the resulting candidate space is narrower and produces fewer accepted edits late in the run.
- C should not replace A/B2 as the main method.
- C is still useful as:
  - a fallback for B2 when no partner source exists;
  - an ablation showing that naive same-row protection is not enough;
  - a building block for D1 if protected candidates are used selectively instead of as the main pool.

Changed files:

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_config_validation.py`
- `tests/test_repairs.py`
- `docs/HANDOFF.md`

Tests / commands run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py::test_protected_same_row_preserves_harmed_query_while_entering_target tests/test_config_validation.py::test_protected_repair_controls_are_allowed`
  - Result: `2 passed`.
- 5-step smoke:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant protected_same_row --run.output_dir outputs/exp_protected_same_row_smoke --qdte.max_iters 5 --qdte.stop_patience 5 --qdte.log_every 1 --qdte.candidate_diagnostics true --evaluation.compute_true_query_error false --evaluation.compute_heldout_query_error false --evaluation.save_synthetic_csv false`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py`
  - Result: `26 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_config_validation.py`
  - Result: `51 passed`.
- 2000-step C run:
  - Output: `outputs/exp_protected_same_row_2000_protected_same_row/`
  - Final measured loss: `11.509982`

Next recommended task:

- Do not spend more time on standalone C before D1.
- Implement D1: bounded exact partner selection.
  - Generate a partner pool per seed, as B2 already does.
  - Score `delta_seed + delta_partner` exactly before attaching/keeping partners.
  - Keep only top positive aggregate partners per seed.
- After D1, optionally add C as a fallback inside B2/D1 only when partner source search fails.

## C/D Constructive Edit Plan - 2026-06-12

Context:

- The accept-count annealing exploration is paused.
- Current implemented constructive methods:
  - A: pool-level `constructive_pair` transport, pairing already-generated candidate edits.
  - B/B2: synthesized/attached partner candidates via `constructive_partner` / `constructive_partner_b2`.
- The next conceptual items from the earlier menu are:
  - C: protected same-row repair.
  - D: exact best partner optimization.

C: protected same-row repair:

- Goal:
  - Given one seed edit `e = x_old -> x_seed`, construct a better `x_new` for the same record.
  - It should still move in the target residual direction.
  - It should explicitly avoid or preserve the harmed queries caused by the seed edit.
- Harm definition:
  - Compute `delta_seed[q] = phi_q(x_seed) - phi_q(x_old)`.
  - Compute `weight[q] = residual[q] * inv_variance[q]`.
  - Harmed queries are `H = {q: delta_seed[q] * weight[q] < 0}`.
- Protection rule:
  - For top harmed queries, prefer `phi_q(x_new) == phi_q(x_old)`, i.e. zero delta on those protected queries.
  - If `weight[q] > 0` and seed exits `q`, repair/protect so the new row still satisfies `q`.
  - If `weight[q] < 0` and seed enters `q`, repair/protect so the new row does not satisfy `q`.
- Practical heuristic:
  - Generate normal single-query seed edits.
  - For each high-harm seed, select top `k` harmed queries by weighted damage.
  - Run several random restarts:
    - apply target repair;
    - apply protection repairs for harmed queries in shuffled or damage order;
    - re-check that the target query still moves in the intended direction;
    - discard unchanged or target-broken candidates.
  - Score final candidates with the unchanged full QDTE edit advantage.
- Suggested compiler:
  - `qdte.candidate_compiler: protected_same_row`
  - aliases: `protected_repair`, `constructive_protected`
  - new repair type: `17`
- Suggested config:
  - `protected_repair_seed_fraction`
  - `protected_repair_harm_queries`
  - `protected_repair_restarts_per_seed`
  - `protected_repair_max_protection_passes`
  - `protected_repair_require_target_direction`
- Why C matters:
  - It directly addresses "one record should be changed into what" without requiring another partner row.
  - It is a more directed version of mask: the protected mask is chosen from measured harmful support, not sampled randomly.
  - It can be B/B2's fallback when no partner source exists.
- Main risk:
  - Target and protection constraints can conflict.
  - Too much protection can reduce candidate diversity or discard many candidates.
  - It should be evaluated with both measured loss and true offline error because it may fit noisy projection more tightly.

D: best partner optimization:

- Full exact D is not recommended as the immediate next implementation.
  - A true exact partner would optimize over source row choice and feasible destination value assignment under many overlapping query predicates.
  - With conjunction/range/halfspace workloads this becomes a small combinatorial/constraint optimization problem per seed.
  - It is likely slower than our intended directed-evolution advantage.
- Recommended D1 instead: bounded exact partner selection.
  - For a seed edit, generate a partner pool using the existing B/B2 source-row and repair logic.
  - Generate more than one partner per seed/harmed query.
  - Compute each partner's delta.
  - Score the aggregate pair exactly:
    - `delta_pair = delta_seed + delta_partner`
    - `adv_pair = delta_pair @ (residual * inv_variance) - 0.5 * ((delta_pair * delta_pair) @ inv_variance) - lambda_cost * pair_cost`
  - Attach only the top positive partner(s) per seed.
- Suggested compiler:
  - `qdte.candidate_compiler: constructive_partner_best`
  - aliases: `constructive_partner_d1`, `best_partner`
  - reuse repair type `16` for attached partners or add `18` if diagnostics need separation.
- Suggested config:
  - `constructive_partner_pool_per_seed`
  - `constructive_partner_best_keep_per_seed`
  - `constructive_partner_best_min_advantage`
  - `constructive_partner_best_delta_backend`
- Why D1 matters:
  - B2 currently constructs attached partners, then leaves aggregate selection mostly to transport.
  - D1 makes partner synthesis itself more selective by using the exact aggregate pair objective before adding/attaching partners.
  - This should reduce low-quality attached partner clutter without claiming a global exact optimum.
- D2 future version:
  - True exact best partner via enumeration, QP/ILP/CP-SAT, or scoped dynamic programming.
  - Keep as future work unless D1/C still plateau and profiling shows partner quality is the bottleneck.

Recommended order:

- Implement C first.
  - It is simpler, cheaper, and more directly tied to the individual-record insight.
  - It also gives a fallback path for B2 when no partner source exists.
- Then implement D1.
  - Reuse B2 partner pool generation and add exact aggregate pre-selection.
  - Avoid D2 for now.
- Evaluation:
  - Compare fixed-8 A, `constructive_pair_accept_anneal32`, B2, C, B2+C if implemented, D1, and random_best_edit.
  - Use at least the 3-seed smoke default.
  - Track diagnostics:
    - protected candidates generated/kept;
    - target-direction failure rate;
    - protected harmed-query delta eliminated;
    - partner pool size;
    - D1 positive aggregate pair rate.

Changed files:

- `docs/HANDOFF.md`

Tests / commands run:

- Read `docs/HANDOFF.md`, `qdte/evolution/candidates.py`, and `tests/test_repairs.py`.
- No code tests were run; this was a planning/documentation update.

Next recommended task:

- Implement C as `protected_same_row` first, with small unit tests proving that a protected repair preserves a harmed query while still moving the target query.

## Constructive Pair Accept-Count Annealing - 2026-06-12

Task:

- Check whether `constructive_pair` would improve if it accepted only one edit, and then test an annealed edit-count schedule.
- Clarify why fixed single-edit `constructive_pair` can be worse than `random_best_edit`.

Implementation:

- `qdte/evolution/engine.py`
  - Added `qdte.accepted_per_iter_schedule`.
  - Supported schedules:
    - `fixed` / `none`
    - `linear`
    - `cosine`
    - `exponential`
  - Added schedule controls:
    - `accepted_per_iter_start`
    - `accepted_per_iter_end`
    - `accepted_per_iter_warmup_iters`
    - `accepted_per_iter_anneal_iters`
  - The schedule only changes the per-iteration transport `max_accept`.
  - QDTE residual, measured loss, delta, and edit-advantage invariants are unchanged.
  - Added `accept_limit` to `metrics_timeseries.csv`.
  - Added accept-schedule fields to `metrics_final.json` and `runtime.json`.
- `scripts/run_ablation.py`
  - Added `constructive_pair_accept_anneal`.
  - Added `constructive_partner_b2_accept_anneal`.
  - Added `constructive_pair_accept_anneal32`.
  - Added `constructive_partner_b2_accept_anneal32`.
  - For constructive pair variants, the default anneal floor is now `2`, not `1`, because a pair unit needs two edits.
  - The `anneal32` variants set `accepted_per_iter=32`, `accepted_per_iter_start=32`, and `accepted_per_iter_end=2`.
- `qdte/config_validation.py`
  - Added validation for schedule names and accept-count parameters.
- Tests:
  - Added config validation coverage for accept schedules.
  - Added a small engine helper test that cosine schedule monotonically anneals to the requested floor.

Why fixed `accepted_per_iter=1` is weak:

- `constructive_pair` units can contain two candidate edits.
- The transport `max_accept` is counted in candidate/edit units.
- Therefore `max_accept=1` prevents accepting any two-edit pair unit.
- So `constructive_pair + accepted_per_iter=1` is not "choose one best pair"; it mostly degenerates into single directed edits.
- `random_best_edit`, by contrast, uses a much broader 100% random candidate pool and then accepts the single best QDTE edit-advantage candidate. This setting is naturally strong when only one edit can be accepted per round.

Seed-0 2000-step smoke comparison:

All rows use `configs/smoke.yaml`, seed `0`, `local_table_feasible_jax`, 2000 iterations, and 512000 scored candidates.

| Method | Final measured loss | True RMSE | True MAE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| A fixed-8 `constructive_pair` | 2.213864 | 0.003275 | 0.002267 | 1535 |
| A fixed-1 `constructive_pair` | 19.054537 | 0.004773 | 0.002716 | 830 |
| A anneal 8->1 `constructive_pair_accept_anneal` | 7.265584 | 0.003948 | 0.002465 | 1202 |
| A anneal 8->2 `constructive_pair_accept_anneal` | 6.853158 | 0.003959 | 0.002498 | 1220 |
| A anneal 32->2 `constructive_pair_accept_anneal32` | 1.932743 | 0.003295 | 0.002243 | 1750 |
| `random_best_edit` | 7.243474 | 0.003833 | 0.002502 | 1175 |
| old `random_mutation` | 65.748090 | 0.003896 | 0.002457 | 1747 |

Milestone losses:

| Method | iter<=100 | iter<=500 | iter<=1000 | iter<=2000 |
| --- | ---: | ---: | ---: | ---: |
| A fixed-8 `constructive_pair` | 39.892043 | 5.923445 | 2.626935 | 2.213864 |
| A anneal 8->2 | 39.892043 | 7.869787 | 6.853158 | 6.853158 |
| A anneal 32->2 | 7.942800 | 2.233513 | 2.044579 | 1.932743 |
| `random_best_edit` | 2078.194516 | 180.407902 | 17.657672 | 7.243474 |

Interpretation:

- Fixed-1 is clearly bad for `constructive_pair` because it disables pair units.
- Annealing helps a lot compared with fixed-1.
- Keeping the floor at `2` is better than floor `1`, consistent with preserving two-edit pair feasibility.
- Small-start annealing (`8->2`) is still worse than fixed-8 on this seed.
- Large-start annealing (`32->2`) is promising:
  - much faster early convergence than fixed-8;
  - lower final measured loss than fixed-8 on seed 0;
  - true RMSE is close to fixed-8, while true MAE is slightly better.
- Current evidence supports running a multi-seed robustness check for `32->2`, but not yet replacing the main algorithm based on a single seed.

Changed files:

- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_config_validation.py`
- `tests/test_engine_smoke.py`
- `docs/HANDOFF.md`

Tests / commands run:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_config_validation.py::test_accepted_per_iter_schedule_controls_are_allowed tests/test_config_validation.py::test_unsupported_or_unknown_config_fails_fast tests/test_engine_smoke.py::test_scheduled_accept_limit_cosine_anneals_to_one`
  - Result: `13 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile scripts/run_ablation.py`
- 2000-step `constructive_pair_accept_anneal` with floor `1`:
  - Output: `outputs/exp_constructive_pair_accept_anneal_2000_constructive_pair_accept_anneal/`
  - Final measured loss: `7.265584`
- 2000-step `constructive_pair_accept_anneal` with floor `2`:
  - Output: `outputs/exp_constructive_pair_accept_anneal2_2000_constructive_pair_accept_anneal/`
  - Final measured loss: `6.853158`
- 2000-step `constructive_pair_accept_anneal32` / override-equivalent run:
  - Output: `outputs/exp_constructive_pair_accept_anneal32_2000_constructive_pair_accept_anneal/`
  - Final measured loss: `1.932743`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile scripts/run_ablation.py qdte/evolution/engine.py qdte/config_validation.py`
- 1-step `constructive_pair_accept_anneal32` sanity:
  - Output: `outputs/exp_constructive_pair_accept_anneal32_sanity_constructive_pair_accept_anneal32/`
  - `accept_limit=32`
  - `accepted=32`
  - `evaluation.compute_true_query_error=false`

Next recommended task:

- Keep fixed-8 `constructive_pair` as the conservative main A method until multi-seed results are available.
- Treat `constructive_pair_accept_anneal32` as the promising next ablation.
- Run a 3-seed robustness comparison for:
  - fixed-8 `constructive_pair`
  - `constructive_pair_accept_anneal32`
  - `constructive_partner_b2`
  - `constructive_partner_b2_accept_anneal32`
  - `random_best_edit`
- Also consider an adaptive accept limit that shrinks only when batch interference is diagnosed, instead of a schedule that always shrinks with iteration.

## Random-Best-Edit, Single-Query, And Strict-Privacy Interpretation - 2026-06-12

Task:

- Clarify what `random_best_edit` corresponds to.
- Compare it with the earlier `single_query` method.
- Explain why `smoke_strict_privacy_1000` can have lower measured loss but higher true RMSE.

Definitions:

- `random_best_edit`
  - Pure random one-row mutation proposal.
  - `random_candidate_fraction=1.0`.
  - It does not use active query residuals to construct the candidate row.
  - It still uses QDTE's exact edit-advantage scoring:
    - `delta @ (residual * inv_variance) - 0.5 * ((delta * delta) @ inv_variance) - lambda_cost * edit_cost`.
  - In the robustness matrix, it used `accepted_per_iter=1`, so it accepts one best improving random edit per iteration.
  - It is a strong random-search / random-best-edit ablation, not exact Private-GSD.
- `single_query`
  - Directed candidate construction.
  - Select active residual queries.
  - If `residual[q] > 0`, it tries to create an enter edit for query `q`.
  - If `residual[q] < 0`, it tries to create an exit edit for query `q`.
  - Then the candidate is still scored by the full QDTE edit advantage, not only by the target query.
  - Standard old setting used `accepted_per_iter=8` and `random_candidate_fraction=0.05`.
- `random_mutation`
  - Different from `random_best_edit`.
  - It uses pure random proposals but the old ablation often used `accepted_per_iter=8`.
  - In older all-strategy runs it was much worse on measured loss than the later `random_best_edit` setting.

Useful existing numbers:

| Run | Variant | Final measured loss | True RMSE | Accepted edits | Scored candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| `outputs/exp_regression_oldbudget2000_single_query` | `single_query` | 5.822247 | 0.003879 | 1461 | 512000 |
| `outputs/robust_ab_20260612/smoke_default_2000/seed0_private_gsd_mutate` | `random_best_edit` alias | 7.243474 | 0.003833 | 1175 | 512000 |
| `outputs/exp_allstrategies2000_random_mutation` | `random_mutation`, 8 accepts/iter | 65.748090 | 0.003896 | 1747 | 512000 |

Interpretation:

- `random_best_edit` is much stronger than the old `random_mutation` ablation on measured loss because it only accepts the best single random edit per iteration.
- `single_query` remains a directed-construction method and is not equivalent to `random_best_edit`.
- In the comparable seed-0 2000-step smoke-style results:
  - `single_query` has better measured loss than `random_best_edit`;
  - true RMSE is very close;
  - `random_best_edit` is useful as a strong random-search baseline, but it does not replace `single_query`.
- The main paper story should not compare only against weak blind/random mutation; `random_best_edit` is a better internal random-search ablation.

Strict privacy explanation:

- `smoke_strict_privacy_1000` sets `privacy.rho_total=0.25` instead of the default `rho_total=1.0`.
- Under zCDP Gaussian noise, reducing rho by 4x roughly doubles the noise standard deviation and quadruples the variance.
- QDTE measured loss is weighted:
  - `measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]`.
- Larger DP variance means smaller `inv_variance`, so the weighted measured loss is not directly comparable across privacy budgets.
- Therefore lower measured loss under strict privacy does not mean a better synthetic dataset.
- The offline true RMSE increases because the projected target is noisier and farther from the real answers.
- Correct comparison:
  - compare A/B2/random within the same privacy setting;
  - do not compare absolute measured loss across `rho_total=1.0` and `rho_total=0.25`.

Changed files:

- `docs/HANDOFF.md`

Tests / commands run:

- Read historical `metrics_final.json` and resolved configs for:
  - `exp_regression_oldbudget2000_single_query`
  - `exp_allstrategies2000_random_mutation`
  - `robust_ab_20260612/smoke_default_2000/seed0_private_gsd_mutate`
- No code tests were run; this was an analysis/documentation update.

Next recommended task:

- Add `single_query` into the corrected robustness matrix if we want a fully fair A/B2/single_query/random_best_edit table under identical seeds and scenarios.

## Robustness Matrix Results After Relabeling - 2026-06-12

Task:

- Explain the completed robustness results after correcting `private_gsd_mutate` to `random_best_edit`.
- Preserve the completed run outputs while preventing the old label from being interpreted as exact Private-GSD.

Outputs:

- Raw completed matrix:
  - `outputs/robust_ab_20260612/robustness_summary.csv`
  - `outputs/robust_ab_20260612/convergence_summary.csv`
- Relabeled copies:
  - `outputs/robust_ab_20260612/robustness_summary_relabelled.csv`
  - `outputs/robust_ab_20260612/convergence_summary_relabelled.csv`
- Completed runs:
  - `33` total run directories with `metrics_final.json`.

Important label correction:

- Rows originally named `private_gsd_mutate` are now interpreted as `random_best_edit`.
- This is a strong random baseline inside QDTE:
  - random one-row mutations;
  - exact QDTE edit-advantage scoring;
  - one best improving edit per iteration.
- It is not exact upstream Private-GSD.

Mean final results:

| Scenario | Variant | Seeds | Final measured loss | True RMSE | True MAE | Accepted edits | Scored candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke_default_2000` | `constructive_pair` | 3 | 5.039149 | 0.003676 | 0.002434 | 1404.3 | 512000 |
| `smoke_default_2000` | `constructive_partner_b2` | 3 | 3.471432 | 0.003570 | 0.002375 | 1431.0 | 543310 |
| `smoke_default_2000` | `random_best_edit` | 3 | 6.462641 | 0.003870 | 0.002519 | 1180.3 | 512000 |
| `smoke_low_candidates_1000` | `constructive_pair` | 3 | 6.874802 | 0.003927 | 0.002552 | 1458.0 | 64000 |
| `smoke_low_candidates_1000` | `constructive_partner_b2` | 3 | 5.633641 | 0.003777 | 0.002469 | 1420.7 | 79985 |
| `smoke_low_candidates_1000` | `random_best_edit` | 3 | 34.179800 | 0.005805 | 0.003039 | 984.7 | 64000 |
| `smoke_strict_privacy_1000` | `constructive_pair` | 3 | 1.911739 | 0.006675 | 0.004612 | 1573.0 | 256000 |
| `smoke_strict_privacy_1000` | `constructive_partner_b2` | 3 | 2.051765 | 0.006675 | 0.004620 | 1507.3 | 271638 |
| `smoke_strict_privacy_1000` | `random_best_edit` | 3 | 5.999340 | 0.007045 | 0.004757 | 988.7 | 256000 |
| `smoke_larger_workload_1000` | `constructive_pair` | 2 | 6.748320 | 0.003191 | 0.002369 | 1417.0 | 181248 |
| `smoke_larger_workload_1000` | `constructive_partner_b2` | 2 | 8.238153 | 0.003221 | 0.002367 | 1296.5 | 232734 |
| `smoke_larger_workload_1000` | `random_best_edit` | 2 | 72.916608 | 0.004862 | 0.003464 | 1000.0 | 256000 |

Convergence mean measured loss:

| Scenario | Iter | `constructive_pair` | `constructive_partner_b2` | `random_best_edit` |
| --- | ---: | ---: | ---: | ---: |
| `smoke_default_2000` | 100 | 45.758 | 37.483 | 2495.793 |
| `smoke_default_2000` | 500 | 7.052 | 4.781 | 220.556 |
| `smoke_default_2000` | 1000 | 5.342 | 3.950 | 16.054 |
| `smoke_default_2000` | 2000 | 5.039 | 3.471 | 6.463 |
| `smoke_low_candidates_1000` | 100 | 447.414 | 403.037 | 2705.303 |
| `smoke_low_candidates_1000` | 500 | 11.652 | 8.783 | 296.758 |
| `smoke_low_candidates_1000` | 1000 | 6.875 | 5.634 | 34.180 |
| `smoke_strict_privacy_1000` | 100 | 15.777 | 12.998 | 638.627 |
| `smoke_strict_privacy_1000` | 500 | 2.089 | 2.496 | 63.730 |
| `smoke_strict_privacy_1000` | 1000 | 1.912 | 2.052 | 5.999 |
| `smoke_larger_workload_1000` | 100 | 237.153 | 201.468 | 4079.154 |
| `smoke_larger_workload_1000` | 500 | 7.131 | 8.539 | 1040.378 |
| `smoke_larger_workload_1000` | 1000 | 6.748 | 8.238 | 72.917 |

Interpretation:

- `constructive_partner_b2` is not a failure:
  - it wins the candidate-limited setting on all 3 seeds;
  - it has faster early convergence in the default and larger-workload settings;
  - it also wins the default 2000-step mean because seed 2 is much better under B2.
- `constructive_pair` remains the more robust final optimizer in:
  - strict privacy after 500/1000 iterations;
  - larger workload after 500/1000 iterations.
- `random_best_edit` is far worse in every scenario despite using exact QDTE advantage scoring.
  - This supports the claim that directed candidate construction / constructive transport matters beyond merely scoring random mutations.
- True RMSE differences between A and B2 are often small compared with measured-loss differences.
  - This reinforces the earlier warning that measured loss can overfit the projected noisy target.

What is trustworthy:

- The A vs B2 comparison is useful.
- The random baseline is useful as a strong QDTE-internal random-best-edit ablation.
- The old `private_gsd_mutate` label is not trustworthy and should not be used in text or plots.

What still needs work:

- Exact upstream Private-GSD comparison is still missing.
- `pgsd_style_mutate50` has only passed a 10-step sanity check:
  - output: `outputs/robust_ab_20260612_check/pgsd50_pgsd_style_mutate50/`
  - final measured loss: `3970.87`
  - candidates scored: `500`
  - accepted edits: `10`
- A full `pgsd_style_mutate50` matrix can be run, but it is still not exact upstream Private-GSD.

Next recommended task:

- Use `constructive_pair` as the stable main algorithm.
- Use `constructive_partner_b2` as a low-budget / early-convergence improvement and ablation.
- Use `random_best_edit` as a strong random-search ablation.
- Decide whether to implement a true upstream Private-GSD wrapper before making paper-level Private-GSD claims.

## Private-GSD Baseline Label Correction - 2026-06-12

Context:

- User correctly questioned the `private_gsd_mutate` robustness result.
- The observed larger-workload result, e.g. final measured loss around `72` after 1000 steps, is too strong to casually call "Private-GSD".

Correction:

- The previous `private_gsd_mutate` variant is not a run of the upstream Private-GSD package.
- It is a QDTE-internal random-best-edit ablation:
  - candidate generation is pure random one-row mutation;
  - each candidate is scored by QDTE's exact edit advantage / global loss drop;
  - one best improving edit is accepted per iteration;
  - initialization, projected measurements, inverse-variance weighting, and DP target handling are all from the QDTE stack.
- Therefore, the existing results under `private_gsd_mutate` should be interpreted as `random_best_edit`, not as an exact Private-GSD baseline.

Code changes:

- `scripts/run_ablation.py`
  - Added explicit variant names:
    - `random_best_edit`
    - `random_best_mutation`
  - Kept `private_gsd_mutate` and `pgsd_mutate` only as backward-compatible aliases for the same QDTE-internal random-best-edit ablation.
  - Added `pgsd_style_mutate50` / `private_gsd_mutate50`:
    - random one-row mutation only;
    - `total_candidates_per_iter=50`;
    - `accepted_per_iter=1`;
    - `kappa_noise=0.0`;
    - still uses the local QDTE workload and measurement stack, so it is closer to mutate-only Private-GSD but still not a drop-in upstream Private-GSD run.
- `scripts/run_robustness_matrix.py`
  - Replaced the default baseline name `private_gsd_mutate` with `random_best_edit`.
  - Added `pgsd_style_mutate50` to the default robustness variants.
  - Updated comments to state that these are QDTE-internal or Private-GSD-style ablations, not exact upstream Private-GSD.

Validation:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile scripts/run_ablation.py scripts/run_robustness_matrix.py`
  - Result: passed.
- 10-step `pgsd_style_mutate50` sanity run:
  - Output: `outputs/robust_ab_20260612_check/pgsd50_pgsd_style_mutate50/`
  - Final measured loss: `3970.87`
  - Candidates scored: `500`
  - Accepted edits: `10`
- 10-step `random_best_edit` sanity run:
  - Output: `outputs/robust_ab_20260612_check/randombest_random_best_edit/`
  - Final measured loss: `3949.04`
  - Candidates scored: `2560`
  - Accepted edits: `10`

External reference:

- Upstream Private-GSD source was cloned to:
  - `/home/qianqiu/my_life/baseline/private-de/private-gsd`
- Current commit inspected:
  - `f6150d7821b9675ce9158b456f1a5cff8bc3b3d6`
- Relevant upstream files inspected:
  - `src/genetic_sd/generator/generator_genetic_sd.py`
  - `src/genetic_sd/generator/mutation_strategies.py`

Current status:

- Completed robustness CSVs under `outputs/robust_ab_20260612/` still contain the old `private_gsd_mutate` label.
- Those rows should be relabeled mentally or in post-processing as `random_best_edit`.
- They must not be used as exact Private-GSD results.

Next recommended task:

- Re-run or post-process the robustness matrix with corrected labels.
- If an exact Private-GSD comparison is required, build a separate wrapper around `/home/qianqiu/my_life/baseline/private-de/private-gsd` or implement a strictly matched population-level GSD baseline with the same workload export/import boundary clearly documented.

## Next-Step Assessment After B2 - 2026-06-12

Context:

- A `constructive_pair` is already very close to the smoke setup's practical optimization limit.
- B2 `constructive_partner_b2` fixed the main B1 design issue and nearly reaches A, but does not beat A on the 2000-step smoke run.
- This is not strong evidence against partner synthesis; it is evidence that the current smoke setting is nearly saturated by A.

Current recommendation:

- Treat A `constructive_pair` as the current main algorithm.
- Do not judge C/D by whether they beat A on the current smoke setting, because that setting is too close to saturation.
- Before spending heavily on C/D, run robustness experiments where A may be less saturated:
  - multiple random seeds;
  - fewer iterations / lower candidate budget;
  - larger or higher-dimensional workloads;
  - stricter privacy budgets / noisier targets;
  - real Adult-style configs rather than only smoke.

Role of C and D:

- C/D are still useful if the paper needs a stronger individual-level directed-edit story:
  - A proves pool-level constructive coordination;
  - B2 proves explicit partner synthesis can be attached without destroying A's seed pool;
  - C/D would test whether a single seed edit can be actively repaired or locally optimized rather than relying on the existing candidate pool.
- But C/D should be framed as stress-regime or ablation improvements, not as necessary smoke-setting improvements.

Suggested next task:

- First run a robustness matrix for A, B2, and the existing baselines.
- If A remains dominant across all regimes, freeze A as the main method and only implement a small C-lite/D-lite as a supporting ablation.
- If A weakens in harder regimes, implement D as bounded local-best/protected repair, because it is more directly connected to the key insight "what should this individual record change into?" than a broad C variant.

Changed files:

- `docs/HANDOFF.md`

Tests / commands run:

- No tests were run; this was a planning/documentation update.

Next recommended task:

- Run the robustness matrix before committing to a larger C/D implementation.

## B2 Attached Constructive Partner Implementation And Result - 2026-06-12

Task:

- Implement the conservative B2 version of constructive partner synthesis.
- Preserve A's original directed candidate pool, add synthesized partners as an attached side pool, and score explicit seed-partner pair units directly.
- Run the 100-step smoke sanity check and the 2000-step smoke comparison.

Code changes:

- `qdte/evolution/candidates.py`
  - Extended `CandidateBatch` with optional `attached_pair_indices`.
  - Added candidate compiler aliases:
    - `constructive_partner_b2`
    - `attached_constructive_partner`
    - `constructive_partner_attached`
  - Added `qdte.constructive_partner_side_budget`.
  - Implemented B2 generation:
    - keep the full A-style directed single-query candidate pool;
    - keep the planned random reserve;
    - append synthesized partners as a bounded side pool;
    - record explicit `(seed_idx, partner_idx)` metadata for direct pair scoring.
  - Added repair type `16` for `constructive_attached_partner`.
  - Added diagnostics:
    - `constructive_attached_partner_candidates`
    - `constructive_attached_pair_units`
- `qdte/evolution/transport.py`
  - Updated `choose_constructive_pair_transport` to consume `attached_pair_indices`.
  - Explicit attached pairs are scored using the original QDTE objective on `delta_seed + delta_partner`.
  - Attached partner rows are excluded from standalone single-unit acceptance and from being seed units.
  - Added explicit-pair diagnostics:
    - `constructive_pair_explicit_pairs_evaluated`
    - `constructive_pair_explicit_positive_pairs`
    - `constructive_pair_explicit_pair_units`
- `qdte/evolution/engine.py`
  - Added repair type name `constructive_attached_partner`.
  - Added B2 diagnostics to runtime and candidate-diagnostics time series.
- `qdte/config_validation.py`
  - Allows the B2 candidate compiler aliases.
  - Validates `qdte.constructive_partner_side_budget >= 0`.
- `scripts/run_ablation.py`
  - Added ablation variant `constructive_partner_b2`.
- `tests/test_transport.py`
  - Added explicit attached-pair transport coverage.
- `tests/test_repairs.py`
  - Added B2 candidate-generation coverage.
- `tests/test_config_validation.py`
  - Added B2 config-validation coverage.

Validation:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/transport.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py tests/test_transport.py tests/test_repairs.py tests/test_config_validation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py::test_constructive_pair_transport_scores_explicit_attached_partner_pair tests/test_repairs.py::test_constructive_partner_b2_preserves_seed_pool_and_attaches_partners tests/test_config_validation.py::test_constructive_partner_b2_side_budget_is_allowed`
  - Result: `3 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_transport.py tests/test_config_validation.py tests/test_delta_index.py`
  - Result: `82 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `134 passed`.

100-step B2 smoke:

- Command variant: `constructive_partner_b2`.
- Output: `outputs/exp_constructiveB2_smoke_constructive_partner_b2/`
- Final measured loss: `34.065774`.
- Accepted edits: `776`.
- Candidates requested: `25600`.
- Candidates scored: `38240`.
- Devices: `cuda:0,cuda:1`.
- This is better than the earlier B1 100-step run (`40.286941`), so preserving A's seed pool fixed most of the naive B budget-composition problem.

2000-step B2 run:

- Command variant: `constructive_partner_b2`.
- Output: `outputs/exp_constructiveB2_2000_constructive_partner_b2/`
- `configs/smoke.yaml`
- `projection.consistency.enabled=true`
- `projection.consistency.method=local_table_feasible_jax`
- `qdte.max_iters=2000`
- `qdte.stop_patience=2000`
- `qdte.candidate_diagnostics=true`

Result:

| Variant | Final measured loss | True RMSE | True MAE | Accepted edits | Candidates scored |
| --- | ---: | ---: | ---: | ---: | ---: |
| A `constructive_pair` | 2.213864 | 0.00327542 | 0.00226749 | 1535 | 512000 |
| B1 `constructive_partner` | 4.678831 | 0.00358782 | 0.00241975 | 1318 | 512000 |
| B2 `constructive_partner_b2` | 2.479907 | 0.00332158 | 0.00226749 | 1470 | 542388 |
| A-mixture `constructive_pair_mixture` | 6.177671 | 0.00392575 | 0.00253086 | 1304 | 512000 |

B2 runtime:

- Wall clock seconds: `119.217749`.
- Generation seconds: `84.771217`.
- Candidate generation seconds: `31.729915`.
- Scoring seconds: `25.686260`.
- Transport seconds: `27.121672`.
- Positive returned candidates: `16714`.
- Selected nonconflicting candidates: `1658`.
- Source filter failure rate: `0.305234`.
- Last explicit pair diagnostics:
  - `last_constructive_pair_explicit_pairs_evaluated=22`
  - `last_constructive_pair_explicit_positive_pairs=0`
  - `last_constructive_pair_explicit_pair_units=0`

Interpretation:

- B2 is a real improvement over B1:
  - measured loss: `2.479907` vs B1's `4.678831`;
  - true RMSE: `0.00332158` vs B1's `0.00358782`.
- B2 is close to A but does not beat A on this smoke run:
  - measured loss: `2.479907` vs A's `2.213864`;
  - true RMSE: `0.00332158` vs A's `0.00327542`.
- B2 also scores more candidates than A (`542388` vs `512000`), so it should not be presented as a same-budget win over A.
- The attached-pair mechanism is directionally credible:
  - B2 removes the main B1 flaw, where synthesized partners cannibalized A's seed budget;
  - the 100-step result improves over B1;
  - the 2000-step result nearly reaches A.
- The remaining issue is partner yield:
  - explicit pair positives are useful early but fade late;
  - by the final iteration, explicit positive pairs are `0`;
  - many side-pool partners do not contribute after residuals become small.
- Current best method on this smoke setup remains A `constructive_pair`.

Trustworthy:

- QDTE objective invariants are preserved: candidate and explicit-pair scoring still use `delta @ (residual * inv_variance) - 0.5 * ((delta * delta) @ inv_variance) - lambda_cost * edit_cost`.
- DP boundary is preserved: active scoring uses noisy/projected targets and variances; true query error is only offline evaluation.
- Full test suite passes after the B2 changes.

Needs attention:

- B2 should be treated as an ablation/prototype, not the current main method, because it does not beat A and uses extra candidate budget.
- The partner synthesis rule is still too weak late in optimization.
- Candidate generation remains CPU-side and is a major part of B2 runtime.
- The next improvement should focus on partner quality or adaptive side-budget gating, not simply increasing fixed partner budget.

Next recommended task:

- Keep A `constructive_pair` as the current strongest baseline.
- Run A vs B2 on multiple seeds if the goal is to decide whether the small A/B2 gap is statistically meaningful.
- For algorithmic progress, implement the next constructive step:
  - either adaptive B2, generating attached partners only when seed-partner predicted advantage is competitive;
  - or C/D bounded local-best/protected repair, where the partner is optimized against aggregate pair advantage rather than only constructed from the top harmed query.

## B Efficiency Patch And 2000-Step Result - 2026-06-12

Task:

- Fix the synthesized-partner B efficiency issue.
- Run a 2000-step smoke comparison against A and prior methods.

Code changes:

- `qdte/evolution/candidates.py`
  - Added batched seed-delta helpers for `constructive_partner`:
    - default `jax` backend using `eval_records_queries_arrays`;
    - optional `sparse_cpu` backend using `QueryDeltaIndex`;
    - optional `dense_cpu` backend matching the old full-query CPU evaluation.
  - Added `qdte.constructive_partner_delta_backend`.
  - Replaced the old per-query CPU loop in B seed-delta computation with the selected backend.
- `qdte/config_validation.py`
  - Validates `qdte.constructive_partner_delta_backend in {jax, sparse_cpu, dense_cpu}`.
- `tests/test_config_validation.py`
  - Added config coverage for all B delta backends.

Validation:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/config_validation.py tests/test_config_validation.py tests/test_repairs.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_config_validation.py tests/test_repairs.py::test_constructive_partner_synthesizes_exit_for_seed_harmed_query`
  - Result: `45 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_transport.py tests/test_config_validation.py tests/test_delta_index.py`
  - Result: `78 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `130 passed`.

Smoke sanity run:

- Output: `outputs/exp_constructiveB_fast_smoke_constructive_partner/`
- 100 iterations.
- Final measured loss: `40.286941`.
- Accepted edits: `790`.
- This matches the earlier B smoke result, so the patch changed the delta-computation path, not the B algorithm semantics.

2000-step B run:

- Command variant: `constructive_partner`.
- Output: `outputs/exp_constructiveB2000_fast_constructive_partner/`
- `configs/smoke.yaml`
- `projection.consistency.enabled=true`
- `projection.consistency.method=local_table_feasible_jax`
- `qdte.max_iters=2000`
- `qdte.stop_patience=2000`
- `qdte.candidate_diagnostics=true`

Result:

| Variant | Final measured loss | True RMSE | True MAE | Accepted edits | Candidates scored |
| --- | ---: | ---: | ---: | ---: | ---: |
| A `constructive_pair` | 2.213864 | 0.00327542 | 0.00226749 | 1535 | 512000 |
| B `constructive_partner` | 4.678831 | 0.00358782 | 0.00241975 | 1318 | 512000 |
| A-mixture `constructive_pair_mixture` | 6.177671 | 0.00392575 | 0.00253086 | 1304 | 512000 |
| old/current `single_query` | 5.822247 | 0.00387935 | not re-read in this task | 1461 | 512000 |

Runtime comparison:

| Variant | Wall seconds | Generation seconds | Candidate generation seconds | Scoring seconds | Transport seconds | Devices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A `constructive_pair` | 50.31 | 15.70 | 1.53 | 1.28 | 12.68 | `cuda:0,cuda:1` |
| B `constructive_partner` | 53.87 | 33.66 | 16.05 | 2.29 | 15.09 | `cpu:0` in this run |

Interpretation:

- B is worse than A in this smoke run:
  - measured loss: `4.678831` vs A's `2.213864`;
  - true RMSE: `0.00358782` vs A's `0.00327542`.
- B is still better than A-mixture and better than old/current single-query on measured loss, but it does not beat A.
- This should not be interpreted as "partner construction is impossible".
- More precise naming:
  - the completed run is B1 / naive `constructive_partner`;
  - it is code-correct with respect to QDTE invariants and the DP boundary;
  - it is not the correct implementation of the intended hypothesis "A plus synthesized partners should improve A".
- The current B1 is not "A plus extra partners"; it replaces roughly half of the directed candidate budget with synthesized partner candidates.
- Diagnostics support this budget-composition explanation:
  - B returned slightly more positive candidates overall (`11468`) than A (`11059`);
  - but B selected/accepted fewer useful nonconflicting edits (`1529` selected / `1318` accepted) than A (`1806` selected / `1535` accepted);
  - B's source-filter failure rate was higher (`0.295`) than A (`0.245`);
  - B spent much more candidate-generation time (`16.05s` vs `1.53s`), even after the seed-delta patch.
- Likely cause:
  - B's synthesized partners are generated from the harmed support of seeds, but the current implementation appends them as separate candidates into the same finite candidate pool;
  - this can dilute A's high-quality single-query seed pool;
  - the constructive-pair transport still has to rediscover useful seed-partner combinations from the mixed pool;
  - therefore B can lose useful A candidates without guaranteeing enough better compensating pairs.

Current status:

- B efficiency is fixed enough to run 2000-step smoke experiments.
- B1 does not beat A under the current candidate-budget composition.
- A remains the best current method on this smoke setup.

Next recommended task:

- Superseded by the B2 section above.
- B2 has now been implemented and tested; the next useful step is adaptive B2 or C/D bounded local-best/protected repair.

## Constructive-Edit Roadmap Decision - 2026-06-12

User proposed the next route:

1. Fix B's efficiency issue.
2. Run B properly and compare against A.
3. Continue along the constructive-edit line through D to understand how far this optimization idea can go.

Current assessment:

- This is the right research route.
- A is already a strong result, but it is pool-limited:
  - it only pairs edits that already happen to be generated in the same candidate pool.
- B is the next necessary check:
  - it tests whether explicitly synthesizing a compensating partner can improve over pool-level pairing.
- Before judging B, its current CPU bottleneck should be fixed:
  - the slow path is full-workload seed-delta computation inside `qdte/evolution/candidates.py`;
  - expected fixes are sparse/JAX batched seed-delta computation or reuse of already computed candidate deltas from scoring/transport.
- After B, continue to C/D, but D should be scoped carefully:
  - avoid a global exact best-partner optimizer over the full database/domain first;
  - implement a bounded local-best partner/protected-repair optimizer that searches a controlled source/destination neighborhood and scores the aggregate edit exactly under the QDTE objective.

Recommended staged plan:

1. B-efficiency patch:
   - move seed-delta computation out of the Python per-seed loop;
   - batch seed deltas using the same delta machinery already used by scorer/transport where possible;
   - preserve the DP boundary and QDTE advantage invariant.
2. B experiment:
   - run 2000-step smoke comparison against A with the same budget;
   - record measured loss, true offline RMSE, accepted edits, candidate counts, positive-pair counts, and runtime.
3. C implementation:
   - add protected same-row repair as B's fallback when no useful partner source exists;
   - preserve target movement while locking or preserving predicates from top harmed queries.
4. D implementation:
   - add bounded local-best partner/protected edit;
   - enumerate or sample a small candidate neighborhood, then choose the exact best aggregate QDTE-advantage edit within that bounded neighborhood.

Why this route is strong for the paper:

- It keeps the innovation centered on individual-level directed evolution rather than generic dataset-level patch mutation.
- It gives a clean progression:
  - A: pool-level constructive pairing;
  - B: synthesized compensating partner;
  - C: protected same-row repair;
  - D: bounded local-best constructive edit.
- Even if A is already close to the smoke noise floor, B/C/D provide enough algorithmic substance and ablation depth to support the contribution.

Changed files for this note:

- `docs/HANDOFF.md`

Tests / commands run for this note:

- No code tests were run; this is a planning/documentation update.

Next recommended task:

- Start with the B-efficiency patch, then run the 2000-step B comparison.

## Noise Floor / Loss Lower-Bound Check - 2026-06-12

Context:

- User asked whether the current A result is close to the theoretical lower bound under the smoke-test DP noise budget.
- Important distinction:
  - The optimization objective `measured_loss = 0.5 * sum_q (target_projected[q] - answer_syn[q])^2 * inv_variance[q]` has mathematical lower bound `0` if the synthetic data can exactly match the projected noisy target.
  - The DP noise/statistical floor is instead the weighted loss between the noisy/projected target and the true answers, not a lower bound on `measured_loss`.

Smoke workload calculation:

- Output inspected: `outputs/exp_constructiveA2000_constructive_pair/`
- Workload size: `m = 243` queries.
- Source dataset size: `n_real = 1000`.
- Raw independent Gaussian noise expected loss against truth: `m / 2 = 121.5`.
- Observed raw noisy-target loss against truth: `123.526127`.
- Observed raw standardized RMS noise: `1.008303`.
- After local feasible projection:
  - projected-target loss against truth: `54.763834`;
  - projected standardized RMS noise: `0.671365`;
  - movement from raw noisy target to projected target: `67.878548`;
  - projection diagnostics weighted objective: `67.878562`.
- Projection diagnostics:
  - independent equality constraints: `91`;
  - active lower-bound constraints: `30`;
  - equality-only effective-dimension rough expected floor: `0.5 * (243 - 91) = 76.0`;
  - with active constraints as a rough heuristic: `0.5 * (243 - 91 - 30) = 61.0`;
  - observed `54.763834` is plausible because this is one realized noise draw and the active feasible projection shrinks the target space.

Family breakdown of projected-target loss against truth:

| Family | Queries | Raw noisy-vs-true loss | Projected-vs-true loss | Raw expected loss |
| --- | ---: | ---: | ---: | ---: |
| `mixed` | 45 | 31.399908 | 7.646384 | 22.5 |
| `oneway` | 26 | 9.212881 | 5.223742 | 13.0 |
| `prefix` | 15 | 8.223231 | 0.496737 | 7.5 |
| `range` | 16 | 7.412409 | 0.375530 | 8.0 |
| `twoway` | 141 | 67.277697 | 41.021442 | 70.5 |

Interpretation:

- A (`constructive_pair`) final measured loss is `2.213864`, far below the projected-target-vs-true noise floor `54.763834`.
- This means A is already fitting the projected noisy target much more closely than the true dataset itself fits that target.
- Therefore, further reducing `measured_loss` alone is unlikely to be a reliable sign of better true accuracy; it can become noisy-target fitting.
- A is still meaningful because its offline true RMSE also improved in this smoke run:
  - A true RMSE: `0.00327542`;
  - old/current `single_query` true RMSE: `0.00387935`.
- Next optimization decisions should use multiple seeds and/or a DP-valid heldout/noisy validation workload instead of judging only by lower `measured_loss`.

Changed files for this note:

- `docs/HANDOFF.md`

Tests / commands run for this note:

- Computed the smoke noise-floor numbers by loading `outputs/exp_constructiveA2000_constructive_pair/config_resolved.yaml`, `measurements.json`, and `queries.json`, then evaluating true workload answers offline.
- No code tests were run, because this task only adds analysis documentation.

Current status:

- The relevant smoke-test statistical floor is approximately `54.76` in measured-loss units after the feasible projection, while A's optimized measured loss is `2.21`.
- The optimization lower bound remains `0`, but that is a lower bound for fitting the noisy/projected target, not for recovering truth.
- Relationship to `true_query_rmse`:
  - `54.763834` is a weighted half-squared loss in count units:
    - `0.5 * sum_q (target_projected[q] - true_answer[q])^2 / variance[q]`.
  - `true_query_rmse` is an unweighted RMSE in rate units:
    - `sqrt(mean_q ((answer_syn[q] / n_syn) - (true_answer[q] / n_real))^2)`.
  - Therefore `54.763834` should not be numerically compared directly with `0.00327542`.
  - For the same pair, projected target vs true, the smoke run values are:
    - weighted loss: `54.763834`;
    - standardized RMS: `sqrt(2 * 54.763834 / 243) = 0.671365`;
    - unweighted count RMSE: `3.340751`;
    - unweighted rate RMSE: `0.00334075`.
  - For A's final synthetic answers:
    - synthetic vs projected target measured loss: `2.213864`;
    - synthetic vs projected target standardized RMS: `0.134986`;
    - synthetic vs projected target unweighted rate RMSE: `0.00103414`;
    - synthetic vs true unweighted rate RMSE: `0.00327542`.
  - In this smoke realization, A's true RMSE is about the same scale as the projected target's own DP-noise RMSE against truth, and slightly smaller on the unweighted RMSE metric.

Next recommended task:

- Run A across multiple random seeds and report both measured loss and offline true query error.
- Add a separate validation workload if we want a DP-valid stopping/selection signal that is less tied to the optimized target.

Additional interpretation:

- This should not be stated as "no algorithm can be better than A".
- More precise statement:
  - under the current smoke measurement budget and workload, A's true RMSE `0.00327542` is already at roughly the projected noisy target's own RMSE against truth `0.00334075`;
  - therefore, on this one run, the remaining error is dominated by the DP measurement noise/projection error scale rather than by failure to fit the measured target.
- It is still possible for another method to beat A on average if it uses additional valid structure:
  - stronger feasible-set projection;
  - better synthetic-data constraints;
  - shrinkage/regularization that improves MSE at the cost of bias;
  - a different query workload or allocation;
  - multiple-seed selection using only DP-valid validation signals.
- What is unlikely to help much is only pushing the same measured objective below A's `2.213864`, because that mostly means fitting the projected noisy target more tightly.

## Constructive Compensating Pair Edits - 2026-06-11

### Synthesized Partner B Implementation - 2026-06-12

Implemented the first synthesized-partner version, B, as a candidate compiler:

- `qdte.candidate_compiler: constructive_partner`
- alias: `synthesized_partner`
- intended transport:
  - `qdte.transport_mode: constructive_pair`
  - `qdte.transport_prefix_strategy: best_advantage`

Implementation behavior:

- Generate a bounded seed set with the existing `single_query` enter/exit generator.
- For each seed edit:
  - compute full-workload seed delta;
  - compute harmed support where `delta_seed[q] * residual[q] * inv_variance[q] < 0`;
  - choose the top harmed queries by weighted damage;
  - synthesize partner candidates in the opposite direction:
    - if seed accidentally enters an overfit query, synthesize an exit partner;
    - if seed accidentally exits an underfit query, synthesize an enter partner;
  - append partner candidates with repair type `15`.
- The final acceptance gate is still the existing aggregate constructive-pair transport, so B does not accept a partner merely because it was synthesized.

New controls:

- `qdte.constructive_partner_seed_fraction`
- `qdte.constructive_partner_harm_queries`
- `qdte.constructive_partner_partners_per_seed`
- `qdte.constructive_partner_source_over_sample_factor`

New diagnostics:

- `constructive_partner_candidates`
- `constructive_partner_seed_candidates`
- `constructive_partner_source_attempts`
- `constructive_partner_source_failures`
- repair type name: `constructive_partner`

Changed files for B:

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

Tests / commands run for B:

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_transport.py tests/test_config_validation.py`
  - Result: `70 passed`.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_partner --run.output_dir outputs/exp_constructiveB_smoke --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 100 --qdte.stop_patience 100 --qdte.log_every 50 --qdte.candidate_diagnostics true --evaluation.compute_true_query_error false --evaluation.compute_heldout_query_error false`
  - Result: passed.
  - Output: `outputs/exp_constructiveB_smoke_constructive_partner/`
  - Final measured loss after 100 iterations: `40.286941`
  - Accepted edits: `790`
  - At iteration 100:
    - `constructive_partner_candidates=121`
    - `constructive_partner_seed_candidates=121`
    - `constructive_partner_source_attempts=213`
    - `constructive_partner_source_failures=92`
    - `constructive_pair_positive_pairs=48`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `126 passed`.

Current B status:

- B1 / naive `constructive_partner` is implemented, smoke-tested, efficiency-patched, and 2000-step tested.
- The 2000-step B1 result is recorded in the top section "B Efficiency Patch And 2000-Step Result - 2026-06-12".
- B1 did not beat A:
  - B1 final measured loss: `4.678831`;
  - B1 final true RMSE: `0.00358782`;
  - A final measured loss: `2.213864`;
  - A final true RMSE: `0.00327542`.
- The old full-workload seed-delta CPU loop was replaced with configurable backends:
  - default `jax`;
  - optional `sparse_cpu`;
  - optional `dense_cpu`.
- B1 is still not the right test of the intended B hypothesis because it replaces part of A's seed candidate budget with partner candidates. The correct next implementation is B2: keep A's seed pool and attach synthesized partners as scored seed-partner units.
- Complexity clarification:
  - A (`constructive_pair`) keeps the same candidate generator as `single_query`; its extra work is transport-side pair matching/scoring within the fixed per-iteration candidate pool.
  - With the current smoke setting (`total_candidates_per_iter=256`, pool about `128`), this extra pair work is bounded and did not prevent 2000-step runs from completing in the same practical regime as other 512k-candidate comparisons.
  - A should be described as a fixed-budget constructive transport over the same candidate pool, not as an unbounded search over the database.
  - B1 (`constructive_partner`) is slower in candidate generation because it synthesizes new partner rows and source-filters them.
  - The next meaningful improvement is the B2 algorithmic composition change, not just more speed work.

### 2000-Step A/B Status - 2026-06-12

Ran the implemented A strategy, pool-level constructive pair, before implementing synthesized partner B.

Setup:

- `configs/smoke.yaml`
- `projection.consistency.enabled=true`
- `projection.consistency.method=local_table_feasible_jax`
- `qdte.max_iters=2000`
- `qdte.stop_patience=2000`
- `qdte.candidate_diagnostics=true`
- `total_candidates_per_iter=256`

New outputs:

- `outputs/exp_constructiveA2000_constructive_pair/`
- `outputs/exp_constructiveA2000_constructive_pair_mixture/`

Results:

| Variant | Final measured loss | True RMSE | Accepted edits | Candidates scored |
| --- | ---: | ---: | ---: | ---: |
| `constructive_pair` | 2.213864 | 0.00327542 | 1535 | 512000 |
| `constructive_pair_mixture` | 6.177671 | 0.00392575 | 1304 | 512000 |
| old/current `single_query` | 5.822247 | 0.00387935 | 1461 | 512000 |
| `enumerated_local` | 6.616871 | 0.00398505 | 1820 | 512000 |
| `masked_single_query` | 6.811406 | 0.00393413 | 1423 | 512000 |
| `random_mutation` | 7.147279 | 0.00397264 | 1756 | 512000 |
| `relaxed_masked_single_query` | 7.416830 | 0.00395967 | 1581 | 512000 |
| fair `qdte_mixture_single_enum` | 6.603438 | 0.00390842 | 1462 | 512000 |
| fair `single_query` | 6.982159 | 0.00391263 | 1413 | 512000 |

Interpretation:

- A is very strong in the old-budget single-query setting:
  - measured loss improves from `5.822247` to `2.213864`;
  - true RMSE improves from `0.00387935` to `0.00327542`.
- The qdte-mixture candidate pool is not automatically better for constructive pairing:
  - `constructive_pair_mixture` ended at `6.177671`, worse than pure `constructive_pair`.
- Diagnostics:
  - `constructive_pair` positive pair total in logged diagnostics: `924`;
  - last logged iteration with positive constructive pairs: `500`;
  - after that, the method still plateaus, so synthesized partner B remains useful to test.

### What Changed

- Implemented a new transport mode:
  - `qdte.transport_mode: constructive_pair`
- The candidate generator still produces individual row edits. The new transport layer then constructs compensating pairs from those individual edits:
  - compute each candidate's query-delta vector `delta_e`;
  - compute residual weights `w = residual * inv_variance`;
  - identify seed edits with positive target/full residual direction but harmful collateral support;
  - build an inverted index keyed by `(query_id, delta_sign)`;
  - for a seed edit, retrieve partner edits with opposite signs on harmed queries;
  - score the aggregate pair by the original QDTE objective on `delta_a + delta_b`;
  - accept only positive-advantage units under row-capacity constraints.
- This keeps the paper's intended individual-level novelty: every proposal remains a per-record directed edit, while the pair layer is only a coordination mechanism for cases where two individually directed edits cancel each other's collateral damage.
- Added diagnostics to `metrics_timeseries.csv` and `runtime.json`:
  - `constructive_pair_pool_candidates`
  - `constructive_pair_seed_candidates`
  - `constructive_pair_pairs_evaluated`
  - `constructive_pair_positive_pairs`
  - `constructive_pair_units`
  - `constructive_pair_single_units`
  - `constructive_pair_pair_units`
  - `constructive_pair_selected_units`
  - `constructive_pair_prefix_units`
  - `constructive_pair_selected_candidates`
  - `constructive_pair_accepted_candidates`
- Added ablation variants:
  - `constructive_pair`
  - `constructive_pair_mixture`

### Changed Files

- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_transport.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests / Commands Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/transport.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py tests/test_config_validation.py`
  - Result: `45 passed`.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant constructive_pair --run.output_dir outputs/exp_constructive_pair_smoke --qdte.max_iters 5 --qdte.stop_patience 5 --qdte.log_every 1 --qdte.candidate_diagnostics true --evaluation.compute_true_query_error false --evaluation.compute_heldout_query_error false`
  - Result: passed.
  - Output: `outputs/exp_constructive_pair_smoke_constructive_pair/`
  - Final measured loss after 5 iterations: `3266.51`
  - Accepted edits: `40`
  - Each iteration accepted 4 pair units / 8 row edits.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `124 passed`.

### Current Status

- The constructive pair mechanism is implemented and wired into the main QDTE loop.
- The unit test explicitly covers the key local-optimum escape case:
  - two individual edits have negative standalone advantage;
  - their combined `delta_a + delta_b` cancels harmful support;
  - the pair has positive QDTE advantage and is accepted.
- The smoke run confirms the transport writes diagnostics and accepts real pair units in the existing DP smoke workflow.
- This is still a first implementation. It constructs pairs only inside the current per-iteration candidate pool; it does not yet generate second-order partner edits beyond the pool.

### Clarification: Relation To Directed Enter/Exit

- `single_query` / directed enter-exit is still the first-order mechanism:
  - positive residual query proposes an enter edit;
  - negative residual query proposes an exit edit;
  - each edit is a per-record directed move.
- `constructive_pair` is a second-order coordination layer over those same individual edits:
  - it does not replace directed enter/exit;
  - it combines two already generated directed edits when their collateral query deltas cancel;
  - the pair is accepted only if the aggregate QDTE objective is positive.
- Therefore these two mechanisms are complementary:
  - directed enter/exit answers "what direction should this row move for a residual query?";
  - constructive pair answers "which two directed row moves can be applied together so collateral damage cancels?".
- The current implementation is the weaker constructive version: it constructs pairs from the existing candidate pool. The stronger future version would generate a seed edit first, inspect its harmful support, and synthesize a partner edit specifically against that support.

### Partner Synthesis Feasibility Note - 2026-06-12

- A first useful version of partner synthesis is feasible and not too hard:
  - generate a seed individual edit `e`;
  - compute harmed support `H = {q: delta_e[q] * residual[q] * inv_variance[q] < 0}`;
  - select the top harmed queries by absolute weighted damage;
  - choose a partner source row that can plausibly move in the opposite direction on those queries;
  - repair that row with existing enter/exit repair primitives;
  - score only the aggregate `delta_e + delta_partner` with the original QDTE objective.
- This is a greedy constructive generator, not an exact optimizer. It should be treated as an incremental next implementation, because it preserves the individual-level edit interpretation while expanding beyond the current pool-only pair matching.
- Exact best-partner synthesis is harder:
  - it asks for the row source and destination that maximize aggregate pair advantage under domain constraints and many overlapping query predicates;
  - with mixed conjunction/range/halfspace queries, this becomes a small constraint optimization problem;
  - exact search can get expensive and should not be the first implementation.
- Recommended implementation target:
  - add a `constructive_partner` candidate compiler or transport submode that synthesizes a bounded number of partner edits per high-damage seed;
  - use only top-`k` harmed queries, random restarts, and existing repair functions;
  - keep full-workload aggregate advantage as the final acceptance gate.
- Important boundary:
  - "constructive" does not mean an O(1) closed-form globally optimal edit;
  - it means directed source selection plus directed destination repair, with the final aggregate QDTE objective still used as the gate;
  - exact compensation may be infeasible if the current synthetic table has no row that can move in the required opposite direction without undoing the seed or violating row capacity.
- If no matching partner source exists in the current synthetic data:
  - for an `enter` compensation, choose any existing source row outside the target query and construct the destination by repair;
  - for an `exit` compensation, a source row satisfying the harmed query is required; if none exists, exact one-edit compensation for that query is impossible under fixed table size;
  - in that case the better fallback is same-row protected repair or multi-edit bundle search, not pretending a partner exists.
- A practical implementation can reduce but not eliminate search:
  - maintain satisfaction bitsets / row lists for active or top harmed queries;
  - sample candidate partner sources from `sat(q)` or `not_sat(q)` sets;
  - synthesize destination values with existing repair primitives;
  - score sparse/full `delta_seed + delta_partner`.
- Conceptual distinction from previous directed enter/exit:
  - old `single_query` chooses one residual query and repairs a row only for that query;
  - previous paired enter/exit tries to jointly move mass from an overfit query region to an underfit query region;
  - partner synthesis starts with one already meaningful individual edit, then either finds a second edit to cancel that seed's collateral damage or rebuilds the same-row destination with explicit protected constraints;
  - therefore it is best viewed as a stronger `single_query`, not merely the old paired query again.
- Relation to mask:
  - mask broadened/relaxed query terms without explicitly optimizing which collateral queries should be protected;
  - protected repair would use the seed edit's measured harmful support to decide which predicates must be avoided or preserved;
  - so it is "mask with a direction": the mask is chosen because it protects known harmed residual-weighted queries, not because it randomly relaxes a target query.
- Algorithm-positioning note:
  - QDTE should not be described as a canonical genetic algorithm if the implementation centers on residual-directed repairs, protected repairs, and constructive partners;
  - it is better described as residual-directed evolutionary local search or residual-directed edit-field optimization over synthetic records;
  - the evolutionary aspect is mutation/evaluation/selection on an evolving synthetic dataset, while the novelty is that mutations are directed by residual and collateral structure rather than random genetic variation;
  - in paper language, avoid overclaiming "genetic algorithm" unless a population, crossover/recombination, and population-level selection are actually implemented.
- Hybrid GA positioning note:
  - adding an outer population loop would make the method a genuine hybrid evolutionary / genetic-search framework;
  - the inner QDTE layer remains the key novelty: random mutation is replaced by residual-directed individual-record edits, protected repairs, and constructive partners;
  - an elitist outer loop that keeps the parent/no-op candidate cannot worsen the measured noisy/projected QDTE objective, but this monotonicity is only for the optimized measured objective, not necessarily for true offline error;
  - the fair comparison should report both measured loss and compute budget, because an outer population loop spends more candidate evaluations;
  - paper framing can be: Private-GSD performs population-level random mutation and selection, while QDTE-GSD performs population-level selection over datasets mutated by a residual-directed edit field.
- If the outer population loop is not implemented:
  - the core innovation remains intact as a residual-directed individual edit field over a single evolving synthetic dataset;
  - the paper should not claim a full genetic algorithm, population-level method, or crossover-style evolutionary framework;
  - the contribution should be framed as a directed mutation/local-search operator that can later be embedded in a Private-GSD-style outer loop;
  - this lowers engineering workload and avoids adding a mostly mechanical outer wrapper, but it also means empirical comparisons must address local-optimum behavior directly.
- Recommended route as of 2026-06-12:
  - prioritize the single-dataset QDTE route first;
  - make directed enter/exit, protected repair, constructive partner edits, random reserve, and fair-budget ablations solid;
  - treat the outer population loop as an extension/future-work or optional final experiment, not as the main implementation target;
  - reason: the outer loop is mostly a known wrapper around selection, while the actual novelty is the residual-directed mutation/edit operator;
  - only implement the outer loop if the single-dataset method still plateaus badly after protected/constructive partner improvements, or if comparison with Private-GSD requires a like-for-like population setting.
- Implementation menu for the next constructive-edit step:
  - A. Pool-level constructive pair, already implemented as `transport_mode=constructive_pair`: match two existing candidate edits whose deltas compensate.
  - B. Synthesized partner edit, recommended next: for a high-collateral seed edit, inspect top harmed queries, search source rows in the required `sat(q)` / `not_sat(q)` sets, repair a partner destination in the opposite direction, and score `delta_seed + delta_partner`.
  - C. Protected same-row repair, recommended after B or as B's fallback: rebuild the seed destination so it still moves toward the target residual query while explicitly avoiding/preserving predicates from harmed queries.
  - D. Exact best partner optimization, not recommended now: solve the best source/destination partner exactly over the query constraints; likely too expensive and not needed for the next experiment.
- Current recommendation: implement B first, with bounded top-`k` harmed queries and random restarts; add C if B often fails because no partner source exists.
- Tests run for this note: not run; documentation-only clarification.

### Next Recommended Task

- Run the real comparison at 1000 or 2000 steps:
  - `single_query`
  - `constructive_pair`
  - `qdte_mixture`
  - `constructive_pair_mixture`
  - old-budget `single_query` as the fallback-random reference
- Use `candidate_diagnostics=true` and inspect:
  - whether positive pair units remain available near the old 5.82 loss plateau;
  - whether accepted pair units reduce the late-stage target-positive/full-negative conflict;
  - whether pair search improves measured loss without relying on accidental fallback random.
- If it still plateaus, the next algorithmic step should be constructive partner generation beyond the existing candidate pool: generate a seed edit first, then synthesize partner edits specifically against the seed's harmful support.

## Fair Candidate Budgets And QDTE Mixture - 2026-06-11

### What Changed

- Implemented explicit candidate-budget controls:
  - `qdte.random_candidate_count`
  - `qdte.directed_candidate_count`
  - `qdte.candidate_shortfall_policy`: `random` or `none`
- Added budget diagnostics:
  - `planned_random_candidates`
  - `fallback_random_candidates`
  - `mixture_random_candidates`
  - `directed_candidate_shortfall`
  - `directed_candidate_budget`
  - `random_candidate_budget`
- Added a new candidate compiler:
  - `qdte.candidate_compiler: qdte_mixture`
  - aliases: `fair_mixture`, `directed_mixture`
- `qdte_mixture` combines:
  - `single_query` proposals;
  - `masked_single_query` proposals;
  - `enumerated_local` proposals;
  - `relaxed_masked_single_query` proposals;
  - plus a separately controlled random reserve.
- Refactored single-query generation into an internal reusable function and fixed masked/relaxed masked sub-budgets so they work correctly inside mixtures.
- Fixed `scripts/run_ablation.py` so CLI overrides win over variant defaults, while preserving the variant-suffixed `run.output_dir` behavior.
- Updated candidate diagnostics summary to support `qdte_mixture`, `--variants`, `--allow-missing`, and exact planned/fallback random columns.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/eval/runtime.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `scripts/summarize_candidate_diagnostics.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests / Commands Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py qdte/eval/runtime.py qdte/config_validation.py scripts/run_ablation.py scripts/summarize_candidate_diagnostics.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `61 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `122 passed`.

### 2000-Step Fair-Budget Smoke Setup

All runs used:

- `projection.consistency.enabled=true`
- `projection.consistency.method=local_table_feasible_jax`
- `qdte.max_iters=2000`
- `qdte.total_candidates_per_iter=256`
- `qdte.directed_candidate_count=224`
- `qdte.random_candidate_count=32`
- `qdte.candidate_shortfall_policy=random`
- `qdte.candidate_diagnostics=true`

Core fair-budget outputs:

- `outputs/exp_fairbudget2000_v2_single_query/`
- `outputs/exp_fairbudget2000_v2_masked_single_query/`
- `outputs/exp_fairbudget2000_v2_relaxed_masked_single_query/`
- `outputs/exp_fairbudget2000_v2_enumerated_local/`
- `outputs/exp_fairbudget2000_v2_qdte_mixture/`

Mixture weight outputs:

- `outputs/exp_fairbudget2000_v3_qdtemix_single_enum_qdte_mixture/`
- `outputs/exp_fairbudget2000_v3_qdtemix_single_heavy_qdte_mixture/`

### Fair-Budget Results

Sorted by final measured loss:

| Variant | Final measured loss | True RMSE | Accepted edits | Directed | Planned random | Fallback random | Directed shortfall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qdte_mixture_single_enum` | 6.603438 | 0.00390842 | 1462 | 425368 | 64000 | 22632 | 22632 |
| `single_query` | 6.982159 | 0.00391263 | 1413 | 354528 | 64000 | 93472 | 93472 |
| `qdte_mixture_single_heavy` | 7.123906 | 0.00388253 | 1431 | 437276 | 64000 | 10724 | 10724 |
| `enumerated_local` | 8.123637 | 0.00416975 | 1942 | 448000 | 64000 | 0 | 0 |
| `relaxed_masked_single_query` | 8.314897 | 0.00411211 | 1574 | 389010 | 64000 | 58990 | 58990 |
| `qdte_mixture_default` | 8.548117 | 0.00396902 | 1512 | 448000 | 64000 | 0 | 0 |
| `masked_single_query` | 9.774205 | 0.00423682 | 1371 | 353798 | 64000 | 94202 | 94202 |

Important clarification:

- The earlier non-fair-budget `single_query` result was still reproduced after the refactor:
  - Output: `outputs/exp_regression_oldbudget2000_single_query/`
  - Final measured loss: `5.822247`
  - True RMSE: `0.00387935`
  - Accepted edits: `1461`
- Therefore the fair-budget `single_query` loss increasing to `6.982159` is not a code regression.
- It comes from changing the candidate distribution:
  - old setting: `random_candidate_fraction=0.05`, but source-filter shortfall caused `170,289` fallback random candidates and only `315,711` directed candidates;
  - fair-budget setting: fixed `32` planned random per iteration, resulting in `64,000` planned random, `93,472` fallback random, and `354,528` directed candidates.
- The old run actually relied on a large amount of fallback random search. The fair-budget run reduced that accidental random reserve and made the comparison more controlled, but it also made plain `single_query` worse on this seed.

Random-reserve interpretation:

- On the current smoke workload, random reserve is beneficial when combined with a high-signal directed strategy.
- Evidence:
  - old `single_query` with large accidental fallback random: loss `5.822247`;
  - fair-budget `single_query` with less random reserve: loss `6.982159`;
  - fair-budget `qdte_mixture_single_enum`: loss `6.603438`;
  - pure `random_mutation` from the earlier active-set run: loss `7.147279`.
- Therefore the best current behavior is not pure random and not pure directed. It is directed search plus enough broad random support.
- `enumerated_local` broad support helps, but it has not yet replaced the value of a large random reserve in the old `single_query` run.
- The next experiment should sweep random reserve levels for `single_query` and `single_enum` instead of assuming lower random is fairer or better:
  - e.g. planned random `32`, `64`, `96`, `128` out of `256`;
  - keep diagnostics for fallback random and directed shortfall;
  - compare against the old implicit-fallback setting.

Mixture component counts:

| Variant | Single | Masked | Enumerated | Relaxed | Fallback random |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qdte_mixture_default` | 202000 | 90000 | 112000 | 44000 | 0 |
| `qdte_mixture_single_enum` | 313368 | 0 | 112000 | 0 | 22632 |
| `qdte_mixture_single_heavy` | 303276 | 46000 | 66000 | 22000 | 10724 |

### Current Judgment

- Item 1 is implemented: candidate budgets are now explicit and diagnosed.
- Item 2 is implemented as a first static mixture compiler.
- The most useful mixture in this smoke run is not the default 45/20/25/10 split. It is the simpler `single + enumerated` mixture:
  - It beats fair-budget `single_query` by measured loss: `6.60` vs `6.98`.
  - It keeps true RMSE essentially comparable: `0.003908` vs `0.003913`.
  - It reduces directed shortfall substantially: `22,632` vs `93,472`.
- Adding masked/relaxed components too heavily still hurts measured loss. This is consistent with earlier diagnostics: relaxed masks broaden support but weaken target alignment.
- The promising direction is therefore:
  - keep `single_query` as the high-signal core;
  - add `enumerated_local` as broad support;
  - keep masked/relaxed as small diagnostic/exploration reserves, not as large mixture components.

### Next Recommended Task

- Turn the static `single_enum` mixture into a named ablation variant or default `qdte_mixture` preset.
- Run multi-seed smoke for:
  - `single_query`
  - `enumerated_local`
  - `qdte_mixture_single_enum`
  - `qdte_mixture_single_heavy`
- Then move the best mixture to Adult/ACS scale.
- Longer-term: make mixture weights adaptive from per-type acceptance rate, target component, collateral component, and directed shortfall.

### Long-Step Sanity Check

After discussing whether 2000 QDTE iterations are too few compared with million-step Private-GSD-style runs, ran a 10000-step old-budget `single_query` check:

- Output: `outputs/exp_long10000_oldbudget_single_query/`
- Command used same old-budget `single_query` setup as the reproduced 2000-step run, except:
  - `qdte.max_iters=10000`
  - `qdte.stop_patience=10000`
  - `qdte.log_every=1000`

Result:

| Run | Iterations | Candidates scored | Accepted edits | Final measured loss | True RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| old-budget `single_query` 2000 | 2000 | 512000 | 1461 | 5.822247 | 0.00387935 |
| old-budget `single_query` 10000 | 10000 | 2560000 | 1461 | 5.822247 | 0.00387935 |

Interpretation:

- In QDTE, "iterations" are not the same unit as mutation proposals. One iteration scores 256 candidates here.
- 10000 QDTE iterations already scored 2.56M candidate edits.
- The 10000-step run found no additional accepted edits beyond the 2000-step run. The logs show long final stretches with `positive=0.000` and `accepted=0`.
- Therefore, for this smoke seed and candidate distribution, simply running more iterations does not improve the current local optimum.
- The next improvement should target proposal support and escape mechanisms, not only more iterations:
  - larger candidate batches per iteration;
  - stronger random reserve sweep;
  - multi-edit or batch proposals;
  - non-greedy/annealed acceptance or population-style dataset proposals;
  - adaptive mixture weights.

### Plateau Diagnosis Versus Population-Level Search

Additional diagnosis for the 10000-step old-budget `single_query` run:

- Last accepted iteration: `1670`.
- Positive candidate rows after iteration `2000`: `0`.
- Accepted candidate rows after iteration `2000`: `0`.
- Last 500 iterations:
  - mean full advantage: `-0.405074`;
  - mean target component: `+0.068672`;
  - mean collateral component: `-0.473219`;
  - target-positive/full-negative rate: `1.000000`;
  - residual conflict rate: `0.986180`;
  - random candidate positive rate: `0.000000`;
  - directed candidate shortfall: about `97.49` candidates/iteration.
- Remaining measured loss is mostly in mixed queries:
  - `mixed`: `4.479680` out of total `5.822247`.

Interpretation:

- The plateau is a single-record local optimum under the current proposal and greedy acceptance rule.
- Directed single-query edits still move the selected target query in the correct direction, but they damage enough other measured queries that the full edit advantage is always negative.
- Random single-record edits also have zero positive rate in the final phase.
- Private-GSD-style population-level search is different: it can evaluate and select whole mutated datasets, so a candidate dataset may improve after several coordinated edits even when each individual edit would be rejected by a greedy one-edit rule.
- Therefore the closest QDTE-side analogue is not merely "run more steps"; it is a macro proposal:
  - generate bundles of `k` edits;
  - sum their deltas;
  - score the bundle by the same QDTE objective;
  - accept the bundle if aggregate edit advantage is positive.
- This preserves the QDTE measured-target DP boundary while giving the search a way to cross valleys that single-edit greedy evolution cannot cross.

Next implementation candidates:

1. Multi-edit bundle transport:
   - sample `k` one-record edits from directed + random proposal pools;
   - compute aggregate `delta_total`;
   - use the same advantage formula with `delta_total`;
   - apply non-conflicting bundles if positive.
2. Population/beam QDTE:
   - keep `B` synthetic datasets;
   - mutate each with several edits;
   - score only against noisy/projected measurements;
   - select the best or use exponential selection.
3. Annealed/epsilon acceptance:
   - occasionally accept small negative one-edit moves early;
   - keep DP-safe because scoring still uses only noisy/projected targets;
   - compare carefully because it changes the optimization semantics.

Important refinement after discussion:

- A generic patch-level search would weaken the QDTE innovation, because Private-GSD-style population search can also mutate a dataset-level patch and select better datasets.
- The core QDTE claim should remain individual-level directed editing:
  - for a current record `x`, use residuals and query structure to infer which values or local edits should move `x` toward reducing the measured objective;
  - score the edit by the full-workload edit advantage;
  - treat the set of candidate edits as a directed per-record vector field over the synthetic dataset.
- Patch or bundle logic should only be a coordination layer over individually directed edits, not the main proposal mechanism.
- The plateau shows that the current individual vector field is often locally correct for the target query but globally rejected because of collateral query damage:
  - final target component is positive;
  - final collateral component is more negative;
  - therefore no single edit has positive full advantage.
- A more faithful next step is "individual directed edit + compensating partner" rather than generic patch mutation:
  - first generate top individual edits for records using the QDTE residual-directed rule;
  - then pair or bundle edits whose collateral deltas cancel each other;
  - accept only if the aggregate full-workload advantage is positive.
- This preserves the original insight, because the algorithm still answers "what direction should this record move?" before any dataset-level coordination.

Constructive compensating-edit idea:

- Randomly hoping two candidate edits cancel each other's collateral damage is likely too low-probability late in optimization.
- The constructive object is the sparse query-delta vector of each individual edit:
  - `delta_e[q] = phi_q(x_new) - phi_q(x_old)`.
- For an edit `e`, define:
  - beneficial support: queries where `delta_e[q] * residual[q] * inv_variance[q] > 0`;
  - harmful support: queries where this quantity is `< 0`.
- A compensating edit `g` for `e` should:
  - have opposite delta signs on the harmful support of `e`;
  - avoid undoing the strongest beneficial support of `e`;
  - maximize aggregate pair advantage `adv(delta_e + delta_g)`.
- The pair objective has an explicit cancellation bonus:
  - `adv(a+b) = adv(a) + adv(b) - sum_q a[q] * b[q] * inv_variance[q]`;
  - if `a[q]` and `b[q]` have opposite signs, the cross term is positive.
- Implementation direction:
  - generate many individual directed edits as usual;
  - compute their sparse deltas;
  - build an inverted index keyed by affected query and delta sign;
  - for a blocked high-target edit, retrieve candidate partners from queries it harms;
  - score the pair exactly by the original QDTE objective before accepting.

## Current Open Work And Improvement Plan - 2026-06-11

### What Changed

- No source code changes.
- Added a status/planning note after the 2000-step relaxed-mask and all-strategy ablations.

### Changed Files

- `docs/HANDOFF.md`

### Tests / Commands Run

- `sed -n '1,80p' docs/HANDOFF.md`
  - Result: inspected current handoff state.
- Unit tests were not run because this task only updates planning documentation.

### Current Judgment

- The implementation now has a credible QDTE core:
  - DP measurement path with noisy/projected targets;
  - edit-advantage objective over a single evolving synthetic dataset;
  - multiple candidate compilers;
  - precise local-table feasible projection via CPU SLSQP and JAX active-set QP;
  - candidate diagnostics and 2000-step ablations.
- The current best smoke strategy under precise local-table projection is `single_query`, not relaxed mask.
- `relaxed_masked_single_query` is useful diagnostically because it reduces target-positive/full-negative conflict, but it weakens target alignment and does not improve final loss as a standalone compiler.
- The most important remaining work is not to add many more ad hoc candidate variants. It is to make the directed proposal distribution stronger while preserving broad support and to validate the result at larger scale.

### Main Missing Pieces

1. Fair candidate-budget controls:
   - Current variants can differ in fallback/random reserve, source-filter shortfall, and generated main-proposal count.
   - Add explicit fixed main proposal budgets and equal random reserve across `single_query`, `masked_single_query`, and `relaxed_masked_single_query`.

2. A principled mixture compiler:
   - Combine high-target-component specialists (`single_query` or `masked_single_query`) with broad support (`enumerated_local` / random reserve).
   - Use relaxed masks as a capped reserve rather than the whole proposal distribution.
   - Learn or schedule mixture weights from diagnostics such as acceptance rate, target component, and collateral component.

3. Better proposal distribution, not just broader support:
   - Relaxed mask broadened support but reduced target signal.
   - Next candidate generator should optimize for expected full-workload advantage, not only target-query satisfaction.
   - A practical direction is residual-weighted local enumeration around a source row: enumerate a small neighborhood, score by full edit advantage, then return top local candidates.

4. Scale validation:
   - Smoke is useful but not enough for a paper claim.
   - Need repeated seeds and Adult/ACS-scale runs with the same DP boundary.
   - JAX active-set projection is dense and may not scale; larger workloads likely need sparse ADMM/PGM or cached/reused projected measurements.

5. Baselines and paper comparison:
   - Implement or wrap a Private-GSD-style population/random-mutation baseline under the same workload/privacy setup.
   - Compare against random mutation with edit advantage, query-directed QDTE, and population-level Private-GSD-style search.
   - The paper claim should separate two ideas:
     - directed edit proposal generation;
     - single-dataset full-workload edit-advantage evolution, as opposed to only population-level selection.

6. Statistical reliability:
   - Current rankings are mostly one-seed smoke results.
   - Need multi-seed tables with confidence intervals or at least mean/std over measured loss and offline true RMSE.

7. Documentation and algorithm definition:
   - The algorithm definition still needs the user's formal writeup.
   - Once finalized, align code names, docs, and experiments with that definition.

### Next Recommended Task

- Implement the fair-budget mixture experiment first:
  - fixed random reserve for all directed variants;
  - fixed main directed proposal budget;
  - mixture of `single_query` / `masked_single_query` + `enumerated_local` + small relaxed-mask reserve;
  - 2000-step smoke ablation under `local_table_feasible_jax`;
  - candidate diagnostics summary comparing target component, collateral component, and acceptance rate.

## Relaxed Mask 2000-Step JAX Active-Set Ablation - 2026-06-11

### What Changed

- No algorithm code changes in this task.
- Ran the requested 2000-step comparison after `local_table_feasible_jax` was upgraded to the precise active-set QP solver.
- Included both normal edit-advantage transport and `blind_accept` controls.
- First ran the mask/random/enumerated/proposal core set, then supplemented the previous paired/exit-only strategy set so the final comparison covers 18 normal variants and 18 blind variants.

### Changed Files / Outputs

- Updated `docs/HANDOFF.md`.
- Generated normal-run outputs:
  - `outputs/exp_jaxactive2000_core_<variant>/`
  - `outputs/logs/exp_jaxactive2000_core_<variant>.log`
- Generated blind-run outputs:
  - `outputs/exp_jaxactive2000_blindcore_blind_<variant>/`
  - `outputs/logs/exp_jaxactive2000_blindcore_blind_<variant>.log`
- Generated extension-run outputs:
  - `outputs/exp_jaxactive2000_ext_<variant>/`
  - `outputs/logs/exp_jaxactive2000_ext_<variant>.log`
  - `outputs/exp_jaxactive2000_blindext_blind_<variant>/`
  - `outputs/logs/exp_jaxactive2000_blindext_blind_<variant>.log`

### Commands Run

- Normal 2000-step core variants:
  - variants: `random_mutation`, `residual_weighted_mutation`, `single_query`, `masked_single_query`, `relaxed_masked_single_query`, `enumerated_local`, `soft_single_query`, `residual_value_mutation`, `proposal_mixture`
  - common overrides:
    - `--projection.consistency.enabled true`
    - `--projection.consistency.method local_table_feasible_jax`
    - `--projection.consistency.max_scope_cells 200000`
    - `--projection.consistency.max_dense_constraint_cells 20000000`
    - `--qdte.max_iters 2000`
    - `--qdte.stop_patience 2000`
    - `--qdte.log_every 200`
    - `--qdte.candidate_diagnostics true`
- Blind 2000-step controls:
  - same variants with `blind_` prefix.
- Extension variants:
  - normal: `paired_query`, `paired_query_full`, `masked_paired_query`, `masked_paired_query_full`, `masked_exit_query`, `masked_exit_query_full`, `directed_exit_only`, `masked_exit_only`, `random_source_directed_exit`
  - blind: same extension variants with `blind_` prefix.
- Candidate diagnostics summary:
  - `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/summarize_candidate_diagnostics.py --base-output outputs/exp_jaxactive2000_core --phase-size 500`
- Unit tests were not rerun in this task because no source code changed.

### Projection Sanity Check

The 2000-step runs used the same JAX active-set projection target:

| Run sample | Method | Solver | Backend | Weighted objective | Final max violation | Solver iterations |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `single_query` | `local_table_feasible_jax` | `jax_dense_active_set_qp` | `gpu` | 67.878562161 | 8.64e-12 | 31 |
| `relaxed_masked_single_query` | `local_table_feasible_jax` | `jax_dense_active_set_qp` | `gpu` | 67.878562161 | 8.64e-12 | 31 |
| `blind_relaxed_masked_single_query` | `local_table_feasible_jax` | `jax_dense_active_set_qp` | `gpu` | 67.878562161 | 8.75e-12 | 31 |

### Normal Edit-Advantage Results

Sorted by final measured loss:

| Variant | Final measured loss | Final true query MAE | Final true query RMSE | Accepted edits | Positive rate | Wall clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_query` | 5.822247 | 0.00248148 | 0.00387935 | 1461 | 0.027102 | 45.23s |
| `enumerated_local` | 6.616871 | 0.00255556 | 0.00398505 | 1820 | 0.022187 | 43.09s |
| `masked_single_query` | 6.811406 | 0.00248971 | 0.00393413 | 1423 | 0.028512 | 45.83s |
| `soft_single_query` | 6.919087 | 0.00253909 | 0.00393622 | 1693 | 0.021910 | 45.91s |
| `random_mutation` | 7.147279 | 0.00249794 | 0.00397264 | 1756 | 0.019332 | 41.53s |
| `residual_value_mutation` | 7.162309 | 0.00256379 | 0.00405264 | 1820 | 0.019813 | 49.79s |
| `relaxed_masked_single_query` | 7.416830 | 0.00253498 | 0.00395967 | 1581 | 0.022516 | 43.79s |
| `proposal_mixture` | 7.886739 | 0.00257613 | 0.00408752 | 1908 | 0.022578 | 47.09s |
| `residual_weighted_mutation` | 8.096253 | 0.00258436 | 0.00414252 | 1899 | 0.021078 | 49.97s |

Full 18-variant normal ranking after paired/exit-only extension:

| Rank | Variant | Final measured loss | Final true query RMSE | Accepted edits |
| ---: | --- | ---: | ---: | ---: |
| 1 | `single_query` | 5.822247 | 0.00387935 | 1461 |
| 2 | `enumerated_local` | 6.616871 | 0.00398505 | 1820 |
| 3 | `masked_single_query` | 6.811406 | 0.00393413 | 1423 |
| 4 | `soft_single_query` | 6.919087 | 0.00393622 | 1693 |
| 5 | `random_mutation` | 7.147279 | 0.00397264 | 1756 |
| 6 | `residual_value_mutation` | 7.162309 | 0.00405264 | 1820 |
| 7 | `paired_query` | 7.305276 | 0.00394197 | 1344 |
| 8 | `masked_exit_query` | 7.346428 | 0.00402359 | 1402 |
| 9 | `relaxed_masked_single_query` | 7.416830 | 0.00395967 | 1581 |
| 10 | `proposal_mixture` | 7.886739 | 0.00408752 | 1908 |
| 11 | `residual_weighted_mutation` | 8.096253 | 0.00414252 | 1899 |
| 12 | `random_source_directed_exit` | 8.519075 | 0.00402155 | 1755 |
| 13 | `masked_paired_query` | 10.262164 | 0.00426441 | 1310 |
| 14 | `masked_exit_only` | 13.494861 | 0.00416234 | 1414 |
| 15 | `masked_exit_query_full` | 19.416237 | 0.00468822 | 1429 |
| 16 | `directed_exit_only` | 21.508989 | 0.00467460 | 1300 |
| 17 | `paired_query_full` | 40.320410 | 0.00602293 | 885 |
| 18 | `masked_paired_query_full` | 105.023597 | 0.00849643 | 843 |

### Candidate Diagnostics

Normal-run main proposal diagnostics:

| Variant | Main positive rate | Main accepted rate | Mean full advantage | Mean target component | Mean collateral component | Target-positive/full-negative rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_query` | 0.041833 | 0.004159 | -0.341003 | 0.172520 | -0.513522 | 0.958167 |
| `masked_single_query` | 0.039762 | 0.003728 | -0.330626 | 0.159014 | -0.489639 | 0.960238 |
| `relaxed_masked_single_query` | 0.027916 | 0.003750 | -0.436799 | 0.082069 | -0.518868 | 0.445594 |
| `random_mutation` | 0.019332 | 0.003430 | -0.444957 | 0.000000 | 0.000000 | 0.000000 |
| `enumerated_local` | 0.022187 | 0.003555 | -0.434116 | -0.006488 | -0.427628 | 0.013936 |

Interpretation:

- `relaxed_masked_single_query` does reduce the specific target-positive/full-negative diagnostic from about `0.96` to `0.45`, so the relaxed mask idea is doing something real.
- But it also cuts the target component roughly in half versus `single_query` and makes the mean full advantage more negative than old `masked_single_query`.
- The broader masked subquery support is therefore not automatically a better proposal distribution. It admits more edits that are weakly aligned with the selected residual and still carry large collateral damage.

### Blind-Accept Controls

Sorted by final measured loss:

| Variant | Final measured loss | Final true query MAE | Final true query RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `blind_masked_single_query` | 194.905663 | 0.00531687 | 0.00835823 | 16000 |
| `blind_single_query` | 387.368563 | 0.00716461 | 0.01130661 | 16000 |
| `blind_relaxed_masked_single_query` | 1837.022080 | 0.01383128 | 0.02441909 | 16000 |
| `blind_soft_single_query` | 2027.200435 | 0.01316872 | 0.02539466 | 16000 |
| `blind_enumerated_local` | 9099.027372 | 0.01862551 | 0.03614739 | 16000 |
| `blind_residual_weighted_mutation` | 10024.368268 | 0.02100823 | 0.03843508 | 16000 |
| `blind_residual_value_mutation` | 10611.991176 | 0.02424691 | 0.04021915 | 16000 |
| `blind_proposal_mixture` | 18475.056826 | 0.03786831 | 0.06220092 | 16000 |
| `blind_random_mutation` | 20998.406288 | 0.04058436 | 0.06585593 | 16000 |

Full 18-variant blind ranking after paired/exit-only extension:

| Rank | Variant | Final measured loss | Final true query RMSE | Accepted edits |
| ---: | --- | ---: | ---: | ---: |
| 1 | `blind_masked_single_query` | 194.905663 | 0.00835823 | 16000 |
| 2 | `blind_paired_query_full` | 218.404721 | 0.01166296 | 16000 |
| 3 | `blind_paired_query` | 260.662649 | 0.01284379 | 16000 |
| 4 | `blind_single_query` | 387.368563 | 0.01130661 | 16000 |
| 5 | `blind_directed_exit_only` | 466.424877 | 0.01107457 | 16000 |
| 6 | `blind_masked_paired_query_full` | 598.761400 | 0.01934770 | 16000 |
| 7 | `blind_masked_paired_query` | 659.836142 | 0.02010047 | 16000 |
| 8 | `blind_masked_exit_query_full` | 1621.916154 | 0.02273193 | 16000 |
| 9 | `blind_masked_exit_only` | 1802.935483 | 0.02502427 | 16000 |
| 10 | `blind_relaxed_masked_single_query` | 1837.022080 | 0.02441909 | 16000 |
| 11 | `blind_soft_single_query` | 2027.200435 | 0.02539466 | 16000 |
| 12 | `blind_masked_exit_query` | 2502.351838 | 0.02414232 | 16000 |
| 13 | `blind_random_source_directed_exit` | 7837.347311 | 0.03551902 | 16000 |
| 14 | `blind_enumerated_local` | 9099.027372 | 0.03614739 | 16000 |
| 15 | `blind_residual_weighted_mutation` | 10024.368268 | 0.03843508 | 16000 |
| 16 | `blind_residual_value_mutation` | 10611.991176 | 0.04021915 | 16000 |
| 17 | `blind_proposal_mixture` | 18475.056826 | 0.06220092 | 16000 |
| 18 | `blind_random_mutation` | 20998.406288 | 0.06585593 | 16000 |

Blind-control interpretation:

- Blind accepting candidate edits is much worse than full edit-advantage transport for every strategy.
- `blind_relaxed_masked_single_query` is far worse than blind `single_query` and blind `masked_single_query`, which confirms that relaxed mask proposals need full-workload advantage filtering.
- This supports the current QDTE direction: the important object is not just directed candidate generation, but directed generation plus full-workload edit advantage on a single evolving dataset.

### Current Status

- The earlier relaxed-mask 100-step smoke did run, but it was only a smoke check.
- The 2000-step comparison is now complete under precise JAX active-set local-table projection for 18 normal variants plus 18 blind controls.
- On this smoke seed, `relaxed_masked_single_query` is not the best strategy:
  - it beats neither `single_query` nor old `masked_single_query`;
  - it also lands slightly behind `random_mutation` by final measured loss.
- The best normal strategy in this run is `single_query`.
- `enumerated_local` remains a strong broad-support baseline and is second by measured loss.
- Paired/exit-only extension strategies do not beat the top single-query/masked/enumerated strategies under this projection.

### Next Recommended Task

- Do not discard relaxed mask, but treat it as a diagnostic/proposal component rather than a standalone replacement.
- Next useful implementation target: a mixture that keeps `single_query` or `masked_single_query` as high-target-component specialists, adds `enumerated_local` for broad local support, and uses relaxed masks only as a capped reserve.
- For a fairer mask study, add explicit candidate budget controls:
  - fixed number of main directed proposals before fallback;
  - fixed random reserve across `single_query`, `masked_single_query`, and `relaxed_masked_single_query`;
  - diagnostics for source-filter shortfall and fallback-repair rate.

## JAX Active-Set Local-Table Projection - 2026-06-11

### What Changed

- Upgraded `projection.consistency.method: local_table_feasible_jax` from the earlier penalty/Adam approximation to a dense active-set QP solver.
- CPU `local_table_feasible_lsq` remains unchanged as the SciPy SLSQP reference.
- The JAX method now solves the same hard-constrained convex QP as the CPU method:

```text
min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
s.t.  T_s >= 0
      sum(T_s) = N
      shared marginals agree across overlapping scopes
```

- Implementation detail:
  - use independent equality rows after the same rank-revealing QR used by the CPU path;
  - maintain an active set of local-table cells fixed at zero;
  - solve equality-constrained KKT least-squares systems in JAX float64;
  - add blocking variables when an unconstrained step would go negative;
  - release active variables when lower-bound multipliers violate KKT.

### Changed Files

- `qdte/measurement/consistency.py`
- `qdte/measurement/measure.py`
- `qdte/config_validation.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests / Commands Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/measurement/consistency.py qdte/measurement/measure.py qdte/config_validation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_consistency_projection.py tests/test_measurement.py tests/test_config_validation.py`
  - Result: `63 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `118 passed`.
- JAX active-set projection audit:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python /tmp/qdte_jax_projection_audit.py`
- 100-step end-to-end smoke:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant relaxed_masked_single_query --run.output_dir outputs/exp_relaxedmask_jaxactive_gpu_smoke --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 100 --qdte.stop_patience 100 --qdte.log_every 100 --qdte.candidate_diagnostics true`

### Projection Audit

Smoke workload, same DP seed:

| Method | Projection time | Target-vs-true RMSE | Weighted objective | Final max violation |
| --- | ---: | ---: | ---: | ---: |
| CPU `local_table_feasible_lsq` | ~100s in this environment | 0.00334076 | 67.878562163 | 1.42e-13 |
| JAX `local_table_feasible_jax` active set | 35.7s audit / 36.3s full run | 0.00334075 | 67.878562161 | 8.64e-12 |

JAX diagnostics from the 100-step run:

- solver: `jax_dense_active_set_qp`
- backend: `gpu`
- solver iterations: `31`
- final max constraint violation: `8.64e-12`
- final RMS constraint violation: `2.02e-12`
- weighted objective: `67.878562161`
- free KKT residual inf-norm: `6.64e-08`
- active lower-bound multiplier min: `3.87e-11`
- active lower-bound cells: `30`

### Current Status

- The JAX version now matches CPU SLSQP target quality and hard-constraint accuracy on smoke.
- It is slower than the earlier penalty approximation, but still about 3x faster than CPU SLSQP in this environment.
- The implementation is dense and suited to smoke/small local-table projections. Adult-scale still likely needs ADMM/PGM or a sparse active-set implementation.

### Next Recommended Task

- Use `local_table_feasible_jax` for fast but strict local-table projection ablations on smoke.
- Run the 2000-step comparison:
  - `single_query`
  - `masked_single_query`
  - `relaxed_masked_single_query`
  - same random reserve variants.
- If projection size grows, implement a sparse/fixed-shape JAX KKT or ADMM solver to avoid recompilation cost from changing active sets.

## JAX GPU Local-Table Projection Added - 2026-06-11

Superseded by the active-set implementation above. This section records the earlier penalty/Adam prototype and its measurements.

### What Changed

- Added a new projection method: `projection.consistency.method: local_table_feasible_jax`.
- Kept the existing CPU method `local_table_feasible_lsq` unchanged.
- The JAX method uses latent local scope tables like the CPU local-table method, but solves with GPU-friendly Adam + per-scope simplex projection:

```text
min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
      + 0.5 * penalty * ||C T - b||^2
s.t.  T_s >= 0
      sum(T_s) = N   exactly by per-scope simplex projection
```

- Shared marginal constraints are handled by a quadratic penalty, so this is an approximate feasible projection. Diagnostics report final constraint violation.
- Added measurement/config/test integration.

### Changed Files

- `qdte/measurement/consistency.py`
- `qdte/measurement/measure.py`
- `qdte/config_validation.py`
- `tests/test_consistency_projection.py`
- `tests/test_measurement.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests / Commands Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/measurement/consistency.py qdte/measurement/measure.py qdte/config_validation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_consistency_projection.py tests/test_measurement.py tests/test_config_validation.py`
  - Result: `63 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `118 passed`.
- JAX projection audit:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python /tmp/qdte_jax_projection_audit.py`
- 100-step end-to-end smoke:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant relaxed_masked_single_query --run.output_dir outputs/exp_relaxedmask_jaxproj_gpu_smoke --projection.consistency.enabled true --projection.consistency.method local_table_feasible_jax --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --qdte.max_iters 100 --qdte.stop_patience 100 --qdte.log_every 100 --qdte.candidate_diagnostics true`

### Projection Audit

Smoke workload with same DP seed, `jax_iterations=20000`, `jax_constraint_penalty=100`, `jax_learning_rate=0.1`:

| Method | Projection time | Target-vs-true RMSE | Weighted objective | Final max violation |
| --- | ---: | ---: | ---: | ---: |
| CPU `local_table_feasible_lsq` | ~100s in this environment | 0.003341 | 67.879 | 1.42e-13 |
| JAX `local_table_feasible_jax` | ~3.3s audit / 3.8s in full run | 0.003696 | 71.38-71.41 | 0.016-0.025 |

Interpretation:

- JAX projection is much faster and runs on GPU (`jax_backend=gpu`, devices `cuda:0`, `cuda:1`).
- CPU SLSQP remains the stricter reference solver.
- JAX penalty projection is close in target quality on smoke, but its shared-marginal consistency is approximate.

### 100-Step Smoke Result

- Output dir: `outputs/exp_relaxedmask_jaxproj_gpu_smoke_relaxed_masked_single_query/`
- JAX devices: `[CudaDevice(id=0), CudaDevice(id=1)]`
- Runtime:
  - wall clock: `4.93s`
  - measurement/projection: `3.81s`
  - generation: `0.75s`
  - scoring: `0.26s`
- Projection diagnostics:
  - method: `local_table_feasible_jax`
  - solver: `jax_projected_gradient_penalty`
  - final max constraint violation: `0.02527`
  - final RMS constraint violation: `0.00593`
  - weighted objective: `71.41342`
- QDTE after 100 steps:
  - final measured loss: `62.68446`
  - final true RMSE: `0.007828`
  - accepted edits: `790`

### Current Status

- The requested GPU projection was added as a new method, not a replacement.
- It is fast enough to remove the repeated 100s CPU projection bottleneck in smoke experiments.
- It should be treated as an approximate projection until we implement ADMM or another solver that enforces shared marginals more exactly.

### Next Recommended Task

- Run 2000-step comparisons with:
  - CPU `local_table_feasible_lsq`
  - JAX `local_table_feasible_jax`
  - `query_space_feasible_lsq`
- If JAX projection quality is sufficient, use it for fast ablations.
- If exact shared consistency matters, implement a JAX ADMM version instead of a penalty-only version.

## Relaxed Mask And GPU Runtime Check - 2026-06-11

### What Changed

- Added a new candidate compiler: `relaxed_masked_single_query`.
- Existing `masked_single_query` behavior is unchanged.
- Added `scripts/run_ablation.py --variant relaxed_masked_single_query`.
- Added diagnostics support for repair type `rtype_relaxed_masked_single`.
- Updated config validation and tests.
- Checked GPU visibility under elevated permissions.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `scripts/summarize_candidate_diagnostics.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests / Commands Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py qdte/config_validation.py scripts/run_ablation.py scripts/summarize_candidate_diagnostics.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `56 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `115 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -c "import jax; print(jax.devices()); print('backend', jax.default_backend())"`
  - Result: `[CudaDevice(id=0), CudaDevice(id=1)]`, backend `gpu`.
- 100-step smoke:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant relaxed_masked_single_query --run.output_dir outputs/exp_relaxedmask_gpu_smoke --projection.consistency.enabled true --projection.consistency.method local_table_feasible_lsq --projection.consistency.max_scope_cells 200000 --projection.consistency.max_dense_constraint_cells 20000000 --projection.consistency.solver_max_iterations 1000 --qdte.max_iters 100 --qdte.stop_patience 100 --qdte.log_every 100 --qdte.candidate_diagnostics true`

### Current Status

- `relaxed_masked_single_query` implements the broader mask semantics:
  - sample a masked subset of target-query terms;
  - for positive residual, source rows only need to fail the masked subquery, and the edit only needs to satisfy the masked subquery;
  - for negative residual, source rows only need to satisfy the masked subquery, and the edit only needs to break the masked subquery;
  - final acceptance still uses the full QDTE edit advantage over all measured queries.
- This differs from existing `masked_single_query`, which protects unmasked terms and still requires full-query enter/exit.
- The new compiler uses repair type `14` and diagnostics prefix `rtype_relaxed_masked_single`.

### GPU / Runtime Diagnosis

- Under the previous restricted tool sandbox, `/dev/nvidia*` was not visible and JAX fell back to CPU with `CUDA_ERROR_NO_DEVICE`.
- Under elevated permissions, `/dev/nvidia0` and `/dev/nvidia1` are visible and JAX reports two CUDA devices.
- The previous experiments were also explicitly run with `CUDA_VISIBLE_DEVICES=` in several commands, which disabled GPU even after JAX could have used it.
- With elevated permissions and no GPU-disabling environment variable, the 100-step smoke log reports:
  - `JAX devices: [CudaDevice(id=0), CudaDevice(id=1)]`
  - runtime `gpu_devices: ["cuda:0", "cuda:1"]`
  - `score_backend: dense_gpu`
- Current `local_table_feasible_lsq` projection is still CPU-bound because it uses SciPy dense QR and SLSQP. GPU access does not make that projection run on GPU.
- In the 100-step relaxed mask smoke run:
  - wall clock: `105.07s`
  - `time_measurement_seconds`: `103.85s`
  - `time_generation_seconds`: `0.82s`
  - `time_scoring_seconds`: `0.31s`
  - candidates scored per second: `31175.97`
- Therefore the current bottleneck for local-table projection experiments is repeated CPU projection during measurement, not the QDTE iteration loop.

### 100-Step Relaxed Mask Smoke Result

- Output dir: `outputs/exp_relaxedmask_gpu_smoke_relaxed_masked_single_query/`
- Final measured loss: `73.476780`
- Final true RMSE: `0.007538`
- Accepted edits: `779`
- Candidate diagnostics show `rtype_relaxed_masked_single_*` fields are present.

### Next Recommended Task

- Run a fair 2000-step comparison on GPU-visible execution:
  - `single_query` with `random_candidate_fraction=0.0`;
  - `masked_single_query`;
  - `relaxed_masked_single_query`;
  - optionally each with the same random reserve.
- To avoid paying the 100s CPU local-table projection cost for every variant, add measurement/projection caching or run all variants from a shared saved measurement target.
- For true GPU projection, replace SciPy SLSQP with a GPU-friendly optimizer such as ADMM/JAX projected updates, or move to a Private-PGM-style marginal fitting backend. Current SciPy projection cannot simply be switched to GPU.

## Masked Single Query Search Space Clarification - 2026-06-11

### What Changed

- No code changed.
- Inspected `qdte/evolution/candidates.py` to compare `single_query` and `masked_single_query` candidate generation after the local-table feasible projection run.

### Changed Files

- `docs/HANDOFF.md`

### Tests / Commands Run

- `rg "masked_single|single_query|mask_min_terms|target_query_ids|repair_type" qdte/evolution/candidates.py qdte/evolution/gpu_candidates.py -n`
- `sed -n '720,920p' qdte/evolution/candidates.py`
- `sed -n '920,1320p' qdte/evolution/candidates.py`
- `sed -n '1,80p' outputs/exp_localtablelsq2000_candidate_diag_summary.tsv`

### Current Status

- `masked_single_query` and `single_query` are not strictly the same proposal space in the current implementation.
- Important semantic distinction:
  - the intended "broad mask" idea would repair only a subset of query terms and allow the final row to satisfy only that masked subquery;
  - the current implementation uses mask as a protected full-query repair, not as a relaxed subquery target.
- For enter edits:
  - `single_query` samples rows that do not satisfy the full target query, then repairs all target-query terms.
  - `masked_single_query` samples a subset of target-query terms, keeps only rows that already satisfy the unmasked terms, then repairs only the masked terms and still requires the final row to satisfy the full target query.
  - With `mask_max_terms=2`, masked enter cannot repair rows that fail more than two target-query terms for 3- or 4-term queries, while single-query enter can.
- For exit edits:
  - the possible one-term break set is closer between the two methods, but the distribution over broken terms differs because masking first chooses a subset and then breaks inside that subset.
- The ablation configs also differ:
  - `single_query` keeps the smoke default `random_candidate_fraction=0.05`;
  - `masked_single_query` sets `random_candidate_fraction=0.0`.
- In the 2000-step `local_table_feasible_lsq` diagnostics, masked proposals slightly reduce collateral damage on average, but also reduce the target component and positive-tail/acceptance behavior:
  - `single_query` main target component: `0.1725`, collateral: `-0.5135`, accepted rate: `0.00416`;
  - `masked_single_query` main target component: `0.1590`, collateral: `-0.4896`, accepted rate: `0.00373`.

### Next Recommended Task

- If we want a fair "same support, different mask scoring" comparison, add a controlled variant that keeps the full-query source pool and samples a full-query repair, then applies mask only to proposal weighting or diagnostics.
- Also run `single_query` with `random_candidate_fraction=0.0` and `masked_single_query` with the same random reserve to separate mask effects from random-reserve effects.

## Local-Table Feasible Projection - 2026-06-11

### What Changed

- Added `projection.consistency.method: local_table_feasible_lsq`.
- The method builds latent nonnegative local tables for every measured query scope and solves one weighted constrained objective against the original noisy measurements:

```text
min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
s.t.  T_s >= 0
      sum(T_s) = N
      shared marginals agree across overlapping scopes
```

- Integrated the method into `measure_real_dataset` and config validation.
- Added tests for local-table feasibility, latent mixed scopes without complete measured cell partitions, measurement integration, and config validation.
- Ran target audit and a 2000-step smoke ablation over five candidate strategies.
- Updated `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md` with the implementation definition, audit, and experiment interpretation.

### Changed Files

- `qdte/measurement/consistency.py`
- `qdte/measurement/measure.py`
- `qdte/config_validation.py`
- `tests/test_consistency_projection.py`
- `tests/test_measurement.py`
- `tests/test_config_validation.py`
- `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
- `docs/HANDOFF.md`

### Generated Outputs

- `outputs/exp_localtablelsq2000_random_mutation/`
- `outputs/exp_localtablelsq2000_single_query/`
- `outputs/exp_localtablelsq2000_masked_single_query/`
- `outputs/exp_localtablelsq2000_residual_weighted_mutation/`
- `outputs/exp_localtablelsq2000_enumerated_local/`
- `outputs/exp_localtablelsq2000_candidate_diag_summary.tsv`
- `outputs/logs/exp_localtablelsq2000_<variant>.log` for all variants except `random_mutation`, whose full stdout was captured by the tool output.

### Tests / Commands Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/measurement/consistency.py qdte/measurement/measure.py qdte/config_validation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_config_validation.py`
  - Result: `34 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_consistency_projection.py`
  - Result: `15 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_measurement.py`
  - Result: `10 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `113 passed`.
- Ran smoke target audit with identical DP noisy measurements across projection methods.
- Ran 2000-step smoke ablation:
  - `random_mutation`
  - `single_query`
  - `masked_single_query`
  - `residual_weighted_mutation`
  - `enumerated_local`
- Summarized diagnostics with:
  - `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/summarize_candidate_diagnostics.py --base-output outputs/exp_localtablelsq2000 --phase-size 500 --output outputs/exp_localtablelsq2000_candidate_diag_summary.tsv`

### Target Audit

Using the same smoke noisy measurements:

| Target version | Target-vs-true RMSE | Target-vs-true MAE | Min | Max | Weighted move from noisy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw noisy target | 0.006508 | 0.004334 | -17.58 | 1002.53 | 0.000 |
| Ordinary simplex/clip projection | 0.006266 | 0.004056 | 0.00 | 1000.00 | 10.511 |
| Old `local_marginal_ipf` | 0.013520 | 0.005610 | 0.00 | 1000.00 | 291.687 |
| Equality-only `query_space_lsq` | 0.004981 | 0.003016 | -17.58 | 1000.00 | 52.188 |
| Feasible `query_space_feasible_lsq` | 0.004724 | 0.002804 | 0.00 | 1000.00 | 56.175 |
| Feasible `local_table_feasible_lsq` | 0.003341 | 0.002271 | 0.00 | 1000.00 | 67.879 |

Local-table feasible diagnostics:

- query scopes: `14`
- latent local-table cells: `265`
- equality constraints: `109`
- independent constraints after QR: `91`
- shared marginal constraints: `95`
- final max constraint violation: `1.42e-13`
- active lower-bound table cells: `23`
- active upper-bound table cells: `0`

### 2000-Step Smoke Results With `local_table_feasible_lsq`

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `single_query` | 5.822277 | 0.002481 | 0.003879 | 1461 |
| `enumerated_local` | 6.616903 | 0.002556 | 0.003985 | 1820 |
| `masked_single_query` | 6.811439 | 0.002490 | 0.003934 | 1423 |
| `random_mutation` | 7.147311 | 0.002498 | 0.003973 | 1756 |
| `residual_weighted_mutation` | 8.096271 | 0.002584 | 0.004143 | 1899 |

### Current Status

- The implementation is wired and tested.
- The new projection reaches the low measured-loss regime we expected from stronger local-table feasibility, while avoiding old IPF's target-vs-true RMSE failure.
- Compared with `query_space_feasible_lsq`, measured loss is much lower:
  - best query-space feasible: `18.865119`;
  - best local-table feasible: `5.822277`.
- On this single seed, `single_query` becomes best by measured loss and true RMSE is comparable across the top variants.
- `masked_single_query` still does not beat `single_query`; local-table target consistency helps the optimization target, but the current mask proposal likely remains too narrow or discards useful edits.
- The method is biased because of nonnegativity constraints and should be reported separately from equality-only `query_space_lsq`.
- Runtime is higher because the solver uses dense QR plus SLSQP over latent local tables. Smoke is fine; Adult-scale likely needs an ADMM/PGM-style solver.
- Nonfatal JAX CUDA plugin warnings appeared because the environment has no CUDA device available; all runs completed on CPU.

### Next Recommended Task

- Run multi-seed smoke for:
  - `query_space_feasible_lsq`
  - `local_table_feasible_lsq`
  - old `local_marginal_ipf`
  - no consistency
- Add a projection-level residual/candidate conflict audit right after initialization, not only after 2000 iterations.
- Improve mask proposals with a random reserve or broader candidate support, because local-table feasible targets alone did not make `masked_single_query` dominate.
- Design a scalable local-table solver for Adult-scale workloads, likely ADMM or a Private-PGM-style marginal fitting backend.

## Feasible Projection Scope And Masked Strategy Diagnosis - 2026-06-11

### What Changed

- No code changed.
- Checked why `query_space_feasible_lsq` did not reach old `local_marginal_ipf` measured loss (`~8-11`) and why `enumerated_local` beats `masked_single_query` under the feasible projection run.

### Changed Files

- `docs/HANDOFF.md`

### Tests / Commands Run

- Inspected projection diagnostics for:
  - `outputs/exp_feasiblelsq2000_enumerated_local/measurements.json`
  - `outputs/exp_conflictdiag2000_consistency_masked_single_query/measurements.json`
  - `outputs/exp_querylsq2000_single_query/measurements.json`
- Summarized `outputs/exp_feasiblelsq2000_candidate_diag_summary.tsv`.
- Summarized `outputs/exp_conflictdiag2000_consistency_candidate_diag_summary.tsv`.

### Current Status

- `query_space_feasible_lsq` is active and numerically satisfies its constraints:
  - method: `query_space_feasible_lsq`;
  - final max constraint violation: `1.71e-13`;
  - target min/max: `0.0 / 1000.0`;
  - constraints: `170`, independent constraints after QR: `98`.
- The reason its measured loss does not reach old IPF's `~8-11` is likely scope strength, not a simple wiring bug:
  - `query_space_feasible_lsq` enforces nonnegative bounded answers plus equalities over existing complete cell partitions only;
  - old `local_marginal_ipf` creates latent local tables for all query scopes, including scopes without complete measured cell partitions;
  - smoke has `14` query scopes but only `10` complete cell-partition scopes;
  - the four latent-only mixed scopes are `(0,2)`, `(1,3)`, `(2,4)`, `(3,4)`.
- Therefore old IPF is much stronger as an optimization-feasibility projection, even though its target-vs-true RMSE is poor. It imposes local-table feasibility beyond the measured query coordinates we constrain in `query_space_feasible_lsq`.
- Under `query_space_feasible_lsq`, `masked_single_query` still wastes most of its proposal budget:

| Variant | Loss | True RMSE | Target-positive/full-negative | Main positive rate | Main accepted rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `masked_single_query` | 21.426 | 0.004004 | 0.969 | 0.0306 | 0.00292 |
| `enumerated_local` | 18.865 | 0.003838 | 0.0247 | 0.0235 | 0.00379 |

- This explains why `enumerated_local` wins measured loss: it has broader local support and far fewer target-positive/full-negative failures. Masked single-query remains too anchored to one target query, and the current feasible projection does not remove enough collateral contradictions.

### Next Recommended Task

- Implement a stronger feasible projection that uses latent local tables for all selected scopes, but solves the correct weighted objective rather than IPF's heuristic averaging.
- Re-run masked strategies after that stronger local-table feasible projection. The hypothesis that mask should win is more likely to be tested fairly under that projection than under the current query-space bounded projection.

## Feasible Biased Query-Space Projection - 2026-06-11

### What Changed

- Implemented a biased feasible projection method:
  - config method: `projection.consistency.method: query_space_feasible_lsq`;
  - objective: weighted distance to noisy measurements;
  - constraints: existing equality consistency `A z=b` plus `0 <= z <= N`;
  - redundant equality constraints are removed with rank-revealing QR before SLSQP.
- Added tests for nonnegativity, total-count constraints, marginal consistency, measurement integration, and config validation.
- Ran target audit and 2000-step QDTE smoke comparison.

### Changed Files

- `qdte/measurement/consistency.py`
- `qdte/measurement/measure.py`
- `qdte/config_validation.py`
- `tests/test_consistency_projection.py`
- `tests/test_measurement.py`
- `tests/test_config_validation.py`
- `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
- `docs/HANDOFF.md`

### Generated Outputs

- `outputs/exp_feasiblelsq2000_<variant>/`
- `outputs/exp_feasiblelsq2000_candidate_diag_summary.tsv`
- `outputs/logs/exp_feasiblelsq2000_<variant>.log`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/measurement/consistency.py qdte/measurement/measure.py qdte/config_validation.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_consistency_projection.py tests/test_measurement.py tests/test_config_validation.py`
  - Result: `55 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `109 passed`.

### Target Audit

Using the same smoke noisy measurements as prior projection audits:

| Target version | Target-vs-true RMSE | Min | Max |
| --- | ---: | ---: | ---: |
| Raw noisy target | 0.006508 | -17.58 | 1002.53 |
| Ordinary simplex/clip projection | 0.006266 | 0.00 | 1000.00 |
| Old `local_marginal_ipf` | 0.013520 | 0.00 | 1000.00 |
| Equality-only `query_space_lsq` | 0.004981 | -17.58 | 1000.00 |
| Feasible `query_space_feasible_lsq` | 0.004724 | 0.00 | 1000.00 |

Feasible projection diagnostics:

- constraints: `170`
- independent constraints after QR: `98`
- redundant constraints: `72`
- final max constraint violation: `1.71e-13`
- lower-bound active answers: `8`
- upper-bound active answers: `2`

### 2000-Step Smoke Results With `query_space_feasible_lsq`

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `enumerated_local` | 18.865119 | 0.00243210 | 0.00383776 | 1942 |
| `residual_weighted_mutation` | 19.382165 | 0.00238272 | 0.00377505 | 1757 |
| `random_mutation` | 20.662170 | 0.00244033 | 0.00391000 | 1790 |
| `single_query` | 20.797150 | 0.00246502 | 0.00391420 | 1573 |
| `masked_single_query` | 21.426172 | 0.00253086 | 0.00400360 | 1459 |

### Current Status

- The feasible projection supports the expert's claim at the target level on this smoke run: target-vs-true RMSE is lower than equality-only `query_space_lsq`.
- It also lowers QDTE measured loss compared with equality-only `query_space_lsq`:
  - best feasible measured loss: `18.865`;
  - best equality-only measured loss: `21.849`.
- Single-seed final synthetic true RMSE is not uniformly better than equality-only projection:
  - best feasible true RMSE: `0.003775`;
  - best equality-only true RMSE: `0.003553`.
- The result suggests feasible projection improves the optimization target and measured loss, but candidate strategy/optimization interaction still matters.
- This method is biased by construction and should be reported separately from the unbiased equality-only projection.

### Next Recommended Task

- Run multi-seed smoke comparing:
  - no consistency;
  - old `local_marginal_ipf`;
  - equality-only `query_space_lsq`;
  - feasible `query_space_feasible_lsq`.
- Add residual-field diagnostics by projection method to measure whether feasible projection actually lowers target-positive/full-negative conflict.
- If feasible projection remains promising, move from query-space bounds to local-table/PGM feasible projection.

## Literature Notes On Nonnegative Consistency Projection - 2026-06-11

### What Changed

- Researched DP literature around consistency projection, nonnegative feasible post-processing, constrained inference, Census TopDown, and Private-PGM-style marginal fitting.
- Added a literature section to `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`.

### Changed Files

- `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a literature/design task.

### Current Status

- The literature supports including this discussion in the paper.
- Relevant lines:
  - Hay et al. on DP histogram consistency post-processing.
  - Zhu et al. on bias/variance effects of feasible-domain post-processing.
  - Fioretto et al. and Census TopDown on constrained optimization for hierarchical/census counts.
  - McKenna et al. on graphical-model/Private-PGM estimation from noisy marginals.
  - Relaxed marginal consistency as the likely scalability path for complex workloads.
- The paper should separate:
  - equality-only unbiased projection as a statistical baseline;
  - nonnegative/feasible projection as a biased but optimization-friendly target.

### Next Recommended Task

- Implement a nonnegative feasible projection baseline and compare it against `query_space_lsq` for both:
  - target-vs-true utility;
  - QDTE residual-field conflict and measured-loss convergence.

## Nonnegativity Versus Unbiasedness Clarification - 2026-06-11

### What Changed

- No code changed.
- Clarified why equality-only projection can output negative counts and why guaranteed nonnegativity is generally incompatible with exact unbiasedness under two-sided Gaussian/discrete-Gaussian noise.
- Updated the expert note with explicit questions about this incompatibility.

### Changed Files

- `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a design clarification.

### Current Status

- `query_space_lsq` can output negative projected answers because it enforces only affine equalities.
- A projection that always outputs nonnegative counts is generally nonlinear/truncated and will introduce bias near the boundary, especially when true counts are near zero.
- Therefore future feasible/local-table/nonnegative projection should be treated as a biased MSE/optimization-feasibility variant, not the unbiased baseline.

### Next Recommended Task

- Ask an expert whether there is any useful restricted-case estimator that is both always nonnegative and exactly unbiased under our noise/discrete-count setting.
- Otherwise, keep two projection families separate:
  - unbiased equality-only target projection;
  - biased feasible target projection for optimization.

## Query-Space LSQ Does Not Guarantee Optimization Consistency - 2026-06-11

### What Changed

- No code changed.
- Clarified that the new equality-only `query_space_lsq` projection improves target fidelity and enforces selected linear equalities, but does not necessarily solve the QDTE residual-field consistency problem.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a design clarification.

### Current Status

- The lower target-vs-true RMSE of `query_space_lsq` is not the same claim as "the residual field is fully consistent for optimization".
- The current `query_space_lsq` projection only enforces constraints that can be constructed from existing complete cell partitions. It does not enforce the full marginal polytope or all possible overlap constraints.
- It also intentionally avoids nonnegativity/simplex constraints to preserve an equality-only affine estimator. Therefore projected answers may not be realizable by any nonnegative table; observed smoke diagnostics include `min_projected_answer=-17.58`.
- If a projected query vector is equality-consistent but not realizable, QDTE may still have a positive loss floor, and local edits may still see target-positive/full-negative conflicts.
- This explains why `query_space_lsq` has higher measured loss than old `local_marginal_ipf`: old IPF may push targets toward an easier realizable local-table representation, while `query_space_lsq` preserves unbiased equality constraints but not full feasibility.

### Next Recommended Task

- Add residual-field consistency diagnostics after projection:
  - count active residual sign contradictions over one-attribute cell moves;
  - compute candidate target-positive/full-negative rates for each projection method;
  - measure lower-bound feasibility gap by projecting onto the span/cone of achievable synthetic query answers where tractable.
- Consider a separate "optimization-feasible projection" variant after the unbiased baseline, possibly with nonnegativity/local-table feasibility, and report it as biased or bias-variance tradeoff rather than unbiased.

## Discrete Gaussian Noise Discussion - 2026-06-11

### What Changed

- No code changed.
- Clarified that discrete Gaussian noise is a credible replacement for continuous Gaussian noise for integer-valued count queries.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a design discussion.

### Current Status

- The relevant reference is Canonne, Kamath, and Steinke, "The Discrete Gaussian for Differential Privacy".
- Discrete Gaussian noise keeps raw noisy count measurements integer-valued before projection, which is more natural and easier to audit for counting queries.
- The later equality-only `query_space_lsq` projection will generally produce real-valued projected targets, which is fine because it is DP post-processing.
- If implemented, the accountant and recorded variance must use the discrete Gaussian theorem/parameterization rather than silently assuming continuous Gaussian variance is identical in all regimes.

### Next Recommended Task

- Add `privacy.noise_distribution: gaussian | discrete_gaussian` and compare target/projection/QDTE utility under the same privacy budget.

## Equality-Only Query-Space Consistency Projection - 2026-06-11

### What Changed

- Implemented the unbiased first-pass consistency projection:
  - config method: `projection.consistency.method: query_space_lsq`;
  - solves equality-only weighted projection in query-answer space;
  - includes known public dataset size `N` as equality constraints for complete partitions;
  - aligns queries to existing complete cell partitions on the same/superset scope;
  - does not apply nonnegativity, clipping, or simplex constraints.
- Kept the old `local_marginal_ipf` method available for comparison.
- Added unit and measurement/config validation tests.
- Updated the expert note with implementation status and initial audit results.
- Ran 2000-step smoke experiments for five candidate strategies under `query_space_lsq`.

### Changed Files

- `qdte/measurement/consistency.py`
- `qdte/measurement/measure.py`
- `qdte/config_validation.py`
- `tests/test_consistency_projection.py`
- `tests/test_measurement.py`
- `tests/test_config_validation.py`
- `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
- `docs/HANDOFF.md`

### Generated Outputs

- `outputs/exp_querylsq2000_<variant>/`
- `outputs/exp_querylsq2000_candidate_diag_summary.tsv`
- `outputs/logs/exp_querylsq2000_<variant>.log`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/measurement/consistency.py qdte/measurement/measure.py qdte/config_validation.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_consistency_projection.py tests/test_measurement.py tests/test_config_validation.py`
  - Result: `50 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `104 passed`.

### Projection Audit

Using the same smoke noisy measurements as the prior consistency audit:

| Target version | Target-vs-true RMSE |
| --- | ---: |
| Raw noisy target | 0.006508 |
| Ordinary simplex/clip projection | 0.006266 |
| Old `local_marginal_ipf` | 0.013520 |
| New `query_space_lsq` | 0.004981 |

`query_space_lsq` diagnostics on smoke:

- complete cell partitions: `10`
- equality constraints: `170`
- constraint nonzeros: `1974`
- initial max constraint violation: `33.90`
- final max constraint violation: `6.52e-7`
- min projected answer: `-17.58`
- max projected answer: `1000.00`

The negative projected answer is expected because this method intentionally avoids nonnegativity constraints to preserve the equality-only affine estimator.

### 2000-Step Smoke Results With `query_space_lsq`

All runs used `configs/smoke.yaml`, seed `0`, `qdte.max_iters=2000`, `qdte.stop_patience=2000`, and `qdte.candidate_diagnostics=true`.

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `single_query` | 21.848564 | 0.00233745 | 0.00355324 | 1682 |
| `random_mutation` | 22.488030 | 0.00231687 | 0.00372899 | 1975 |
| `masked_single_query` | 22.841380 | 0.00234979 | 0.00368346 | 1542 |
| `residual_weighted_mutation` | 23.731912 | 0.00229630 | 0.00359126 | 1923 |
| `enumerated_local` | 24.286752 | 0.00239506 | 0.00375374 | 2023 |

### Current Status

- The new equality-only projection fixes the major failure mode of old `local_marginal_ipf` on smoke:
  - measured loss is much lower than no-consistency runs (`~22` vs `~64-66`);
  - offline true RMSE improves slightly relative to no-consistency runs (`~0.00355-0.00375` vs `~0.00380-0.00392`);
  - it does not show the old IPF target distortion (`0.0135` target RMSE).
- `single_query` becomes best by measured loss and offline true RMSE under `query_space_lsq`, while `residual_weighted_mutation` has the best true MAE.
- The single-query collateral conflict is reduced but not eliminated:
  - `single_query` target-positive/full-negative rate remains `96.7%`;
  - `masked_single_query` remains `96.6%`.
- Because equality-only projection can produce negative projected counts, the measured objective may contain infeasible targets for any synthetic table. This is intentional for the unbiased baseline, but should be tracked.

### Next Recommended Task

- Run multi-seed smoke for `query_space_lsq` against no-consistency and old `local_marginal_ipf`.
- Add projection diagnostics by query family:
  - target-vs-true offline error by family;
  - count of negative projected answers by family;
  - constraint participation per query.
- Then test Adult-scale feasibility with `query_space_lsq` and inspect constraint count/solver time before considering ADMM.

## Expert Note For Unbiased Consistency Projection - 2026-06-11

### What Changed

- Added a standalone expert-discussion document for consistency projection:
  - project background and why QDTE needs consistency projection;
  - evidence that current `local_marginal_ipf` is problematic;
  - equality-only unbiased weighted projection formulation;
  - how known public dataset size `N` should enter as affine constraints;
  - full-joint/local-scope feasibility limits;
  - staged implementation plan and expert questions.

### Changed Files

- `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md`
- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a documentation/design task.

### Current Status

- The note explicitly recommends avoiding nonnegativity/simplex constraints in the first corrected estimator to preserve unbiasedness.
- The known dataset size `N` is treated as public metadata and should be included directly in the equality system `A z = b`, not as a separate heuristic pre-step.
- The document clarifies that `N` constraints and cross-scope consistency constraints should be solved jointly in the same weighted projection.

### Next Recommended Task

- Review `docs/CONSISTENCY_PROJECTION_EXPERT_NOTE.md` with an expert.
- If the formulation is approved, implement Stage 1: equality-only query-space weighted projection with sparse constraints.

## Consistency Projection Scope Limits - 2026-06-11

### What Changed

- No code changed.
- Clarified the feasibility and statistical properties of corrected consistency projection when full joint tables are infeasible.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a design discussion.

### Current Status

- Exact full global consistency generally requires a full joint table over all attributes, which is infeasible for realistic tabular domains.
- If a query scope itself spans too many attributes/cells, an exact local-table representation for that query is also infeasible. There is no free exact projection in that case unless the query family has special structure that can be represented compactly.
- Practical options for high-scope queries:
  - exclude or cap scopes whose cell count exceeds `max_scope_cells`;
  - keep high-scope query answers as standalone measured variables with weaker consistency constraints;
  - use a reduced set of maximal clique scopes;
  - use approximate structured representations such as sparse supports, low-rank/factorized tables, or decision-diagram style query masks;
  - treat them as held-out/evaluation queries rather than enforced measurement constraints.
- Unbiasedness depends on the estimator objective, not on whether the optimizer is QP or ADMM:
  - equality-only weighted projection onto a linear affine consistency space containing the true answer vector is linear and unbiased under zero-mean noise;
  - adding nonnegativity/simplex inequality constraints makes the estimator nonlinear and can introduce finite-sample bias, often trading bias for variance reduction;
  - ADMM is only an optimizer. If it converges to the same convex objective, it has the same statistical target as the QP. Early stopping or wrong penalties can add optimization bias.
- For complex scope hypergraphs, the most realistic implementation plan is not a full joint table. It is a sparse local-scope consistency problem with a configurable scope budget and a solver that enforces only selected shared marginals.

### Next Recommended Task

- Implement the corrected projection in stages:
  - first equality-only query-space projection on a small, explicitly constructed consistency matrix `C`;
  - then local-table sparse QP on smoke scopes under a strict `max_scope_cells`;
  - then ADMM only if direct sparse QP is too slow.
- Add diagnostics that report:
  - which query scopes were enforced;
  - which high-scope queries were left unconstrained or weakly constrained;
  - number of variables, equality constraints, and active inequality constraints;
  - target-vs-true offline error by enforced and unenforced query family.

## Global Weighted Consistency Projection Design Note - 2026-06-11

### What Changed

- No code changed.
- Organized the corrected consistency-projection problem for implementation/expert discussion.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a design discussion.

### Current Status

- The mathematically intended projection is:
  - minimize `0.5 * sum_q (z_q - y_q)^2 / variance_q`;
  - subject to `z` being induced by nonnegative marginal tables that satisfy known total count and shared-marginal consistency.
- A full-domain table over all attributes is usually infeasible because the joint domain is the product of all attribute cardinalities.
- A tractable formulation should use local scope/clique tables:
  - one table per query scope or workload clique;
  - query answers are linear sums over the table for that scope;
  - tables are constrained nonnegative and sum to `N`;
  - overlapping tables must agree on shared marginals.
- This is a convex quadratic program with linear equalities and nonnegativity constraints. For smoke-scale workloads it can be solved directly. For Adult-scale workloads it likely needs a sparse solver, ADMM, Dykstra-style projection, or a carefully scoped clique graph.
- The current `local_marginal_ipf` implementation is not this objective. It fits local tables and then reconciles derived marginals by scope-level averaging, which can drift away from the original noisy measurement objective.

### Open Design Questions

- Which consistency set should be enforced:
  - all observed local scopes only;
  - a reduced set of maximal clique scopes;
  - only complete partitions and nested prefix/range constraints;
  - or a junction-tree-compatible subset of scopes.
- How to estimate post-projection uncertainty:
  - keep original diagonal variances only as a heuristic;
  - compute projected covariance for query answers;
  - or inflate/regularize variances conservatively after projection.
- Whether nonnegativity/simplex constraints are required in the first corrected version, or whether a linear equality-only weighted projection should be implemented first as a lower-risk baseline.

### Next Recommended Task

- Prototype the corrected projection on smoke using a sparse convex formulation:
  - variables: local table cells for each scope;
  - objective: weighted residual to original noisy queries;
  - constraints: local table totals, nonnegativity, and shared marginal equalities.
- Compare against:
  - ordinary simplex/clip projection;
  - per-scope LSQ without cross-scope reconciliation;
  - current `local_marginal_ipf`.
- If direct sparse QP is too slow, switch to ADMM where each scope solves a local weighted table fit and shared marginals are dual-coupled, while retaining the original noisy objective in every iteration.

## Consistency Projection Culprit Audit - 2026-06-11

### What Changed

- No code changed.
- Audited why current `projection.consistency.enabled=true` lowers measured loss but worsens offline true-query RMSE on the smoke workload.

### Changed Files

- `docs/HANDOFF.md`

### Tests / Commands Run

- Offline diagnostic command using `outputs/exp_conflictdiag2000_random_mutation/` measurements and exact answers for evaluation only:
  - compared `target_noisy`, ordinary `project_targets`, current `project_consistent_targets`, preprojected input to `project_consistent_targets`, and per-scope LSQ with cross-scope IPF disabled through direct function call.

### Current Status

- The current consistency projection behavior is problematic, and the likely culprit is now isolated.
- Target-vs-true RMSE on smoke:
  - raw noisy target: `0.006508`
  - ordinary simplex/clip projection: `0.006266`
  - current full consistency projection: `0.013520`
  - consistency projection with preprojected input: `0.013522`
  - per-scope weighted LSQ with cross-scope IPF disabled: `0.004891`
- Therefore the major distortion is not caused by passing raw noisy targets into consistency projection. It is caused by the cross-scope IPF/reconciliation step in `_enforce_scope_consistency`.
- The implementation is not an exact global weighted projection of noisy query answers onto the full consistency constraint set. It first fits one local table per scope, then iteratively averages shared marginals across scopes. During that iterative reconciliation it is no longer minimizing weighted distance to the original noisy measurements.
- The current scope precision is `sum(inv_variance[q])` over queries in that scope. This can overweight scopes with many correlated/overlapping query constraints and push shared marginals away from more reliable local estimates.
- Keeping diagonal variances after projection is also an approximation. Once targets are projected, query errors become correlated; the measured loss can become easier to reduce even if the projected target is less faithful to the true workload.

### Next Recommended Task

- Do not treat current `projection.consistency.enabled=true` as the main fix for `single_query` yet.
- Add a controlled projection ablation:
  - `local_scope_lsq` / no cross-scope IPF;
  - current `local_marginal_ipf`;
  - ordinary simplex/clip projection.
- Then implement or prototype a real weighted consistency projection:
  - minimize weighted distance to original noisy measurements;
  - enforce nonnegativity and known-total constraints;
  - enforce shared marginal equalities as constraints, not by post-hoc averaging;
  - report conservative post-projection uncertainty instead of pretending the original diagonal variance is unchanged.

## Candidate Conflict Diagnostics And Consistency Check - 2026-06-11

### What Changed

- Added optional per-iteration candidate diagnostics behind `qdte.candidate_diagnostics`.
- The diagnostics are written to `candidate_diagnostics_timeseries.csv` and do not affect active optimization.
- Added `scripts/summarize_candidate_diagnostics.py` to aggregate candidate diagnostics across variants.
- Added smoke-test coverage that verifies candidate diagnostics are emitted.
- Ran 2000-step smoke diagnostics for five strategies with and without `projection.consistency.enabled=true`.

### Changed Files

- `qdte/evolution/engine.py`
- `scripts/summarize_candidate_diagnostics.py`
- `tests/test_engine_smoke.py`
- `docs/HANDOFF.md`

### Generated Outputs

- `outputs/exp_conflictdiag2000_<variant>/`
- `outputs/exp_conflictdiag2000_candidate_diag_summary.tsv`
- `outputs/exp_conflictdiag2000_consistency_<variant>/`
- `outputs/exp_conflictdiag2000_consistency_candidate_diag_summary.tsv`
- `outputs/logs/exp_conflictdiag2000_<variant>.log`
- `outputs/logs/exp_conflictdiag2000_consistency_<variant>.log`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/engine.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile scripts/summarize_candidate_diagnostics.py qdte/evolution/engine.py`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `50 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_engine_smoke.py::test_engine_smoke_outputs`
  - Result: `1 passed`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `97 passed`.

### Experiment Setup

- Base config: `configs/smoke.yaml`
- Variants:
  - `random_mutation`
  - `single_query`
  - `masked_single_query`
  - `residual_weighted_mutation`
  - `enumerated_local`
- Common overrides:
  - `qdte.max_iters=2000`
  - `qdte.stop_patience=2000`
  - `qdte.log_every=2000`
  - `qdte.candidate_diagnostics=true`

### Main Diagnostic Result

| Variant | Loss | True RMSE | Accepted | Main target comp. | Main collateral comp. | Target-positive but full-negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_mutation` | 65.748090 | 0.003896 | 1747 | 0.000000 | 0.000000 | 0.000000 |
| `single_query` | 66.653234 | 0.003801 | 1491 | 0.271707 | -0.686195 | 0.973147 |
| `masked_single_query` | 65.901815 | 0.003811 | 1505 | 0.287948 | -0.673895 | 0.971430 |
| `residual_weighted_mutation` | 63.991130 | 0.003817 | 1911 | -0.003311 | -0.458138 | 0.041018 |
| `enumerated_local` | 64.867884 | 0.003923 | 1974 | -0.003991 | -0.439663 | 0.034670 |

Interpretation:

- The hypothesis is supported. `single_query` and `masked_single_query` almost always generate candidates that are beneficial for their target query, but about `97%` of those target-positive candidates are still full-workload negative.
- The reason is visible in the decomposition:
  - `single_query`: target component `+0.2717`, collateral component `-0.6862`;
  - `masked_single_query`: target component `+0.2879`, collateral component `-0.6739`.
- In the last 500 iterations, both `single_query` and `masked_single_query` reach essentially `100%` target-positive/full-negative candidates. That explains why they stall: they keep producing locally correct edits that the full edit-advantage scorer must reject.
- Residual conflict is common for almost all one-record edits, including random mutation. The distinguishing issue for `single_query` is not merely that conflicts exist, but that its proposal budget is heavily concentrated on target-correct edits whose collateral damage dominates.

### Consistency Projection Check

With `projection.consistency.enabled=true`, measured loss becomes much smaller, but offline true-query error becomes much worse on this smoke run:

| Variant | Consistency loss | Consistency true RMSE |
| --- | ---: | ---: |
| `masked_single_query` | 8.873327 | 0.013201 |
| `random_mutation` | 10.235393 | 0.013371 |
| `single_query` | 10.317552 | 0.013078 |
| `residual_weighted_mutation` | 10.764914 | 0.013319 |
| `enumerated_local` | 10.937878 | 0.013381 |

Offline target audit:

- Without consistency projection, `target_projected` versus exact true query answers has RMSE `0.006266`.
- With current consistency projection, `target_projected` versus exact true query answers has RMSE `0.013520`.
- Therefore current consistency projection makes the noisy target easier/self-consistent for measured loss, but less faithful to the true workload in this smoke setting.
- This exact true audit is evaluation-only and was not used in optimization.

### Current Status

- The single-query underperformance hypothesis is now empirically supported.
- Current consistency projection is not yet a safe answer to the problem. It reduces measured objective conflict but appears to distort projected targets on smoke.
- The most credible current algorithmic direction remains broad directed proposals, especially `residual_weighted_mutation`, plus full edit-advantage scoring.
- `single_query` / `masked_single_query` remain useful as interpretable specialists but should not be the main proposal family without a fix for target-collateral contradiction.

### Next Recommended Task

- Audit consistency projection before using it as a required preprocessing layer:
  - compare projecting raw noisy targets versus first applying partition/simplex and prefix clipping;
  - inspect weighting by variances and the current diagonal post-projection variance approximation;
  - add per-family target-vs-true offline diagnostics for projection experiments;
  - run multi-seed smoke because this is currently one seed.
- Add accepted-candidate diagnostics, not just generated-candidate diagnostics, so we can quantify whether random/residual-weighted wins by finding larger global-positive edits or by avoiding row conflicts.
- After the projection audit, rerun `single_query`, `masked_single_query`, `random_mutation`, and `residual_weighted_mutation` under the corrected projection setting.

## Single-Query Underperformance Hypothesis - 2026-06-11

### What Changed

- No algorithm code changed.
- Documented the current hypothesis for why `single_query` / `masked_single_query` can underperform broad random mutation despite being query-directed.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm-design discussion.

### Current Status

- A single record edit can change many workload query answers at once.
- `single_query` proposes edits that are locally correct for one target residual, but those edits can have negative collateral deltas on other measured queries.
- DP noise and projection can make residual directions locally inconsistent: overlapping predicates on the same attribute or cell can have mixed positive/negative residuals.
- The dense edit-advantage scorer does filter these candidates, but candidate budget is finite. A narrow proposal family can waste much of its budget on candidates that are target-correct but globally weak or redundant.
- `random_mutation` can beat `single_query` because it keeps broad one-edit support and lets the full workload scorer select globally positive edits.
- `masked_single_query` reduces over-constrained repairs but remains anchored to one target query/mask, so it can still miss globally useful local edits.

### Next Recommended Task

- Add diagnostics for target-query benefit versus full-workload advantage:
  - target-only advantage;
  - full advantage;
  - collateral advantage gap;
  - number of affected queries per candidate;
  - residual sign conflict over changed attributes.
- Use these diagnostics to verify whether `single_query` candidates have higher target benefit but worse collateral damage than `random_mutation` and `residual_weighted_mutation`.

## Candidate Generation Explanation - 2026-06-11

### What Changed

- No algorithm code changed.
- Documented the current candidate generation mechanism for QDTE discussion.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a code-reading and explanation task.

### Current Status

- `generate_candidates` builds a fixed-size batch of local record edits `(row_id, old_row, new_row, target_query_id, repair_type, edit_cost)`.
- Candidate generation proposes edits only; the actual acceptance decision is still made later by full measured-objective edit advantage.
- The main proposal families are:
  - `random_mutation`: random row, random mutable attribute, random different value.
  - `single_query`: choose active query by residual sign; hard source filter; repair enter/exit for that query.
  - `masked_single_query`: single-query repair with partial predicate masks.
  - `paired_query` / masked variants: pair negative-residual source queries with positive-residual destination queries.
  - exit-only variants: only generate exits for overfit queries.
  - `residual_weighted_mutation`: random source row; attribute and value distributions are biased by residual-weighted query incidence.
  - `enumerated_local`: sample rows and enumerate one-attribute replacement neighbors.
  - `soft_single_query`: single-query repair with probabilistic source preference.
  - `proposal_mixture`: fixed mixture of random, residual-weighted, enumerated-local, soft-single, and residual-value proposals.
- DP boundary remains clean: proposal distributions use noisy/projected residuals, inverse variances, query definitions, and synthetic data; exact true answers are evaluation-only.

### Next Recommended Task

- Add per-`repair_type` positive/selected/accepted attribution so we can explain why `enumerated_local` accepts more edits while `residual_weighted_mutation` gets lower measured loss.

## 2000-Step All-Strategy And Blind Ablation - 2026-06-10

### What Changed

- No algorithm code changed.
- Ran all current candidate-proposal strategies for 2000 iterations with advantage-gated transport and with `blind_accept`.
- Wrote a combined result table to `outputs/exp_allstrategies2000_summary.tsv`.

### Changed Files

- `docs/HANDOFF.md`

### Generated Outputs

- `outputs/exp_allstrategies2000_summary.tsv`
- `outputs/exp_allstrategies2000_<variant>/`
- `outputs/exp_allstrategies2000_blind_<variant>/`
- `outputs/logs/exp_allstrategies2000_<variant>.log`
- `outputs/logs/exp_allstrategies2000_blind_<variant>.log`
- `outputs/logs/exp_allstrategies2000_status.tsv`

### Tests Run

- No unit tests were run in this experiment-only turn.
- Experiment command pattern:
  - `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py --config configs/smoke.yaml --variant <variant> --run.output_dir outputs/exp_allstrategies2000 --qdte.max_iters 2000 --qdte.stop_patience 2000`
- All 34 runs completed successfully.
- Every run completed `2000` iterations and scored `512000` candidates.

### 2000-Step Results

| Variant | Gated loss | Gated true MAE | Gated true RMSE | Gated accepted | Blind loss | Blind true RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_weighted_mutation` | 63.991130 | 0.00243210 | 0.00381733 | 1911 | 10652.326866 | 0.03901530 |
| `enumerated_local` | 64.867884 | 0.00251440 | 0.00392261 | 1974 | 9293.405088 | 0.03642242 |
| `random_mutation` | 65.748090 | 0.00245679 | 0.00389629 | 1747 | 21052.697981 | 0.06589720 |
| `masked_single_query` | 65.901815 | 0.00242387 | 0.00381086 | 1505 | 348.034071 | 0.00882617 |
| `residual_value_mutation` | 66.356769 | 0.00252263 | 0.00405771 | 1781 | 10108.635935 | 0.03866204 |
| `soft_single_query` | 66.409377 | 0.00247737 | 0.00393361 | 1661 | 1963.638265 | 0.02676533 |
| `single_query` | 66.653234 | 0.00243621 | 0.00380058 | 1491 | 363.587701 | 0.00922490 |
| `random_source_directed_exit` | 66.719440 | 0.00250617 | 0.00385061 | 1809 | 8152.630402 | 0.03618005 |
| `proposal_mixture` | 68.892344 | 0.00252263 | 0.00419926 | 1744 | 18634.290135 | 0.06074107 |
| `masked_paired_query` | 69.634863 | 0.00253498 | 0.00415244 | 1447 | 917.715297 | 0.02244408 |
| `paired_query` | 74.945066 | 0.00276132 | 0.00456818 | 1399 | 336.371147 | 0.01221161 |
| `masked_exit_only` | 75.646192 | 0.00281893 | 0.00449188 | 1503 | 2138.391017 | 0.02370472 |
| `masked_exit_query` | 77.418970 | 0.00267901 | 0.00452565 | 1343 | 1930.821971 | 0.02461673 |
| `masked_exit_query_full` | 79.634513 | 0.00286008 | 0.00463437 | 1635 | 1977.662527 | 0.02622411 |
| `directed_exit_only` | 80.424615 | 0.00279424 | 0.00466623 | 1331 | 718.719381 | 0.01186984 |
| `masked_paired_query_full` | 86.918137 | 0.00292593 | 0.00536219 | 1372 | 802.917081 | 0.02162684 |
| `paired_query_full` | 193.221520 | 0.00468724 | 0.00973539 | 630 | 343.355062 | 0.01299826 |

### Current Status

- The 1000-step result was not just a one-off: `residual_weighted_mutation` remains the best measured-loss strategy at 2000 steps.
- `enumerated_local` accepts the most edits (`1974`) and is second-best by measured loss, but it does not beat `residual_weighted_mutation`; it appears to accept more small improvements rather than better average improvements.
- `random_mutation`, `residual_value_mutation`, `proposal_mixture`, `paired_query`, and several weaker variants did not materially improve after 1000 steps despite running to 2000 iterations. They appear to stall under the current candidate budget.
- Offline true metrics are more nuanced:
  - best true RMSE is `single_query` (`0.00380058`);
  - `masked_single_query` is close (`0.00381086`);
  - `residual_weighted_mutation` is close behind (`0.00381733`) and has the best measured loss.
- Blind loss confirms the advantage gate is essential. Blind broad-mutation proposals are catastrophic, while blind single/paired query repairs are much less bad but still far worse than gated runs.
- The current fixed `proposal_mixture` remains weak. A useful mixture probably needs adaptive budget allocation and per-`repair_type` selected/accepted attribution.

### Next Recommended Task

- Run multi-seed 2000-step smoke for the top candidates:
  - `residual_weighted_mutation`;
  - `enumerated_local`;
  - `masked_single_query`;
  - `single_query`;
  - `random_mutation`.
- Add per-`repair_type` positive/selected/accepted attribution to explain why `enumerated_local` accepts more edits but loses on measured loss.
- Then test top candidates on Adult scale with held-out workload evaluation to separate measured-loss fitting from true-query generalization.

## Broad Directed Proposal Ablation - 2026-06-10

### What Changed

- Added five CPU candidate compilers for broad/support-preserving proposal tests:
  - `residual_weighted_mutation`: random source row; attribute and value sampled from residual/variance-weighted query incidence.
  - `enumerated_local`: sampled source rows; enumerate one-attribute replacement neighbors, ordered by residual value scores.
  - `soft_single_query`: existing single-query enter/exit repair, but source filtering is probabilistic instead of hard.
  - `residual_value_mutation`: random source row and random attribute; value sampled from residual-weighted value scores.
  - `proposal_mixture`: fixed-budget mixture of random reserve, residual-weighted, enumerated-local, soft-single, and residual-value proposals.
- Added residual-weighted value tables using only query definitions, synthetic records, noisy/projected residuals, and inverse variances.
- Added diagnostics columns to `metrics_timeseries.csv` for the new proposal families.
- Changed candidate diagnostics so `random_candidates` is counted by `repair_type == 0`, not by `target_query_id < 0`; broad directed proposals can now carry a reference target id while still being non-random.
- Added ablation runner variants for the five compilers.
- Added candidate/compiler tests and config-validation coverage.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests Run

- `pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result in default shell: failed because `pytest` is not on `PATH`.
- `python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py scripts/run_ablation.py qdte/config_validation.py`
  - Result in default shell: failed because `python` is not on `PATH`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python -m py_compile qdte/evolution/candidates.py qdte/evolution/engine.py scripts/run_ablation.py qdte/config_validation.py`
  - Result: passed.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `50 passed in 0.09s`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `97 passed in 7.00s`.

### 1000-Step Smoke Results

All runs used `configs/smoke.yaml`, seed `0`, `privacy.mode=dp`, `qdte.max_iters=1000`, `qdte.stop_patience=1000`, `total_candidates_per_iter=256`, and `accepted_per_iter=8`. Exact true answers were used only for final offline evaluation metrics.

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits | Candidates scored |
| --- | ---: | ---: | ---: | ---: | ---: |
| `residual_weighted_mutation` | 64.388322 | 0.00241564 | 0.00383239 | 1897 | 256000 |
| `enumerated_local` | 65.282284 | 0.00253498 | 0.00397730 | 1963 | 256000 |
| `residual_value_mutation` | 66.356769 | 0.00252263 | 0.00405771 | 1781 | 256000 |
| `soft_single_query` | 68.739577 | 0.00254733 | 0.00406683 | 1593 | 256000 |
| `proposal_mixture` | 68.892344 | 0.00252263 | 0.00419926 | 1744 | 256000 |

Reference 1000-step runs from the previous ablation:

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `random_mutation` | 65.748090 | 0.00245679 | 0.00389629 | 1747 |
| `single_query` | 68.296449 | 0.00252675 | 0.00403687 | 1512 |
| `masked_single_query` | 66.800469 | 0.00251852 | 0.00400000 | 1486 |

### Current Status

- The strongest new result is `residual_weighted_mutation`: it beats the previous `random_mutation` baseline on both measured loss and offline true-query MAE/RMSE in this single-seed smoke run.
- This supports the current hypothesis: QDTE needs broad local proposal support, but the proposal distribution can still be residual-directed.
- `enumerated_local` also improves measured loss over random, but its offline true metrics are worse than residual-weighted mutation; enumerating neighbors helps coverage but is less targeted.
- `residual_value_mutation` underperforms `residual_weighted_mutation`, which suggests attribute selection carries important signal, not just value selection.
- `soft_single_query` remains weak. Softening the source filter alone does not fix the narrowness of query-repair proposals.
- The current fixed `proposal_mixture` is worse than the best component because it spends too much budget on weak components. A mixture should be adaptive or heavily biased toward the productive broad residual-weighted family.
- A first parallel experiment attempt failed with CUDA OOM because five JAX processes competed for GPU memory. Sequential reruns with `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false` completed successfully.

### Privacy Boundary

- Active optimization still uses only noisy/projected measurements, inverse variances, synthetic records, query definitions, candidate deltas, and edit costs.
- Exact true answers are computed only for offline final evaluation metrics and were not used for active query selection, candidate generation, scoring, transport, stopping, or hyperparameter selection.

### Next Recommended Task

- Promote `residual_weighted_mutation` to the main broad-directed proposal candidate for the paper narrative.
- Run multi-seed smoke and Adult-scale sweeps for `residual_weighted_mutation` against `random_mutation`, `single_query`, and `masked_single_query`.
- Tune:
  - `residual_attr_uniform_mix`;
  - `residual_value_uniform_mix`;
  - `residual_value_temperature`;
  - candidate budget size.
- Add accepted-positive attribution by `repair_type`; final loss is enough for coarse comparison, but adaptive proposal scheduling needs per-family positive/selected/accepted yield.
- Revisit `proposal_mixture` only after attribution is available, with most budget assigned to `residual_weighted_mutation` and a small random support reserve.

## Proposal Support Strategy Discussion - 2026-06-10

### What Changed

- No algorithm code changed.
- Added interpretation of "random reserve" and non-random ways to preserve broad candidate support.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm-design discussion.

### Current Status

- "Keep random reserve" should not be interpreted as "only random can work."
- It means QDTE needs a support-safety component: when a directed proposal family narrows the search space, it must not exclude the small local edits that the full measured objective would accept.
- A directed proposal can beat random if it keeps broad support while increasing the density of high-advantage candidates.

### Candidate Strategies Beyond Pure Random

- Support-dominating mixture:
  - keep a small proposal component with full one-edit support;
  - make the rest residual-directed;
  - this gives a formal safety property that any random one-edit candidate still has nonzero probability.
- Residual-weighted broad mutation:
  - still mutate one or a few attributes, but choose row/attribute/value using query residual weights;
  - unlike current strict source filtering, it does not require the row to satisfy one target query first.
- Enumerated local neighborhood:
  - for selected rows, enumerate all one-attribute replacements or top-k value replacements;
  - score all with exact edit advantage;
  - this is less random and can cover more useful moves per selected row, but may be expensive on high-cardinality columns.
- Delta-sign screening:
  - use a cheap approximate score from changed-attribute query incidence to prefilter many local edits;
  - then score the surviving candidates exactly with the current QDTE advantage.
- Transport-aware paired generation:
  - generate exits and destinations jointly, but keep a broad destination set rather than forcing one source/destination query pair;
  - sample destinations from positive residual mass with diversity constraints.
- Adaptive proposal mixture:
  - track generated/positive/selected/accepted counts by `repair_type`;
  - increase budget for productive families and decay budget for families that collapse.

### Current Judgement

- The next goal should be "directed proposals with full-enough support," not "replace random with a narrower deterministic compiler."
- The most promising non-random path is residual-weighted broad mutation plus exact advantage scoring:
  - row selection biased toward rows that contribute to overfit queries or near underfit regions;
  - attribute selection biased by residual-weighted query incidence;
  - value proposal biased toward entering positive residual predicates and exiting negative residual predicates;
  - no hard source filter unless it is known to preserve support.

### Next Recommended Task

- Implement a new broad directed proposal family, tentatively `residual_weighted_mutation`, and compare it against `random_mutation`, `single_query`, and `masked_single_query`.
- Add per-`repair_type` attribution first if possible, so the next scheduler can learn proposal productivity rather than rely on final loss only.

## Candidate Support Analysis - 2026-06-10

### What Changed

- No algorithm code changed.
- Added interpretation of why `random_mutation + edit_advantage` is strong while masked/paired directed proposals underperform.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a read-only analysis of existing 1000-step outputs.

### Current Status

- `random_mutation` is not "pure random search" in the effective algorithm. It is broad random local proposal plus the same full measured-objective edit advantage and transport gate.
- On the 1000-step smoke runs, this broad proposal pool gives the scorer more independent small moves to choose from.
- Directed/masked/paired proposals have residual signal, but their support is narrower:
  - `single_query`: source filter failure rate `0.1539`;
  - `masked_single_query`: `0.1526`;
  - `paired_query`: `0.1772`;
  - `masked_paired_query`: `0.1729`;
  - `masked_exit_query`: `0.4201`;
  - `masked_exit_query_full`: `0.4887`;
  - `random_mutation`: `0.0`.
- Positive-candidate rate alone is misleading:
  - `directed_exit_only` has positive candidate rate `0.0616`, higher than random's `0.0393`, but worse final loss.
  - Its selected-per-positive ratio is only `0.1101`, versus random's `0.2067`.
  - This suggests many directed positives are redundant, row-conflicting, or become unattractive once full-query/batch effects are considered.
- Full-budget paired variants confirm over-concentration:
  - `paired_query_full` final loss `193.222`, much worse than `paired_query` at `74.945`;
  - `masked_paired_query_full` final loss `87.123`, worse than `masked_paired_query` at `71.171`.
- Blind-accept results show directed proposals do contain signal, but not enough without objective gating:
  - `blind_single_query` loss `310.742`;
  - `blind_masked_single_query` loss `363.256`;
  - `blind_random_mutation` loss `16246.446`;
  - all are still far worse than advantage-gated runs.

### Current Judgement

- The current best explanation is proposal support, not the edit-advantage formula.
- Random mutation wins on this smoke setup because it is a high-coverage local proposal distribution and edit advantage acts as a strong filter.
- Masked/paired variants underperform because they spend too much budget in constrained regions:
  - source rows must satisfy specific query or mask conditions;
  - paired variants require compatible positive/negative query pairs;
  - full-budget paired/masked variants repeatedly propose similar local moves;
  - exit-only variants remove mass but do not construct good destinations.
- The useful algorithmic direction is therefore not replacing random with one directed compiler. It is a proposal mixture:
  - broad random reserve for coverage;
  - single/masked-single for early directed local improvements;
  - paired/masked-paired as minority destination-aware moves;
  - adaptive budget updates by accepted contribution.

### Next Recommended Task

- Add per-`repair_type` attribution:
  - candidates generated;
  - positive advantages;
  - selected;
  - accepted;
  - realized batch contribution.
- Implement a mixture scheduler that keeps random support and adjusts directed proposal budgets online.

## Masked Single-Query Implementation And All-Strategy 1000-Step Runs - 2026-06-10

### What Changed

- Implemented CPU-only `qdte.candidate_compiler: masked_single_query`.
- Added alias `masked_single`.
- Added `repair_type=9`.
- Added diagnostics and timeseries column:
  - `masked_single_query_candidates`
- Added `scripts/run_ablation.py --variant masked_single_query`.
- Existing `blind_` prefix support now also works for `blind_masked_single_query`.
- Updated README and architecture docs.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `40 passed in 0.08s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `87 passed in 6.93s`
- `git diff --check`
  - Result: passed

### Current Status

- `masked_single_query` is a target-preserving masked version of default one-query repair.
- For `residual[q] > 0` enter:
  - sample a subset of ordinary terms as the editable mask;
  - sample near-miss rows that satisfy the unmasked terms and fail at least one masked term;
  - repair only masked terms;
  - keep only candidates that enter the full target query.
- For `residual[q] < 0` exit:
  - sample rows satisfying the full query;
  - break one masked term;
  - keep only candidates that exit the full target query.
- Final scoring and transport are unchanged: accepted edits still optimize the full measured objective.

### 1000-Step All-Strategy Runs

All runs used `configs/smoke.yaml`, seed 0, DP mode, 243 measured queries, 1000 rows, `qdte.max_iters=1000`, `qdte.stop_patience=1000`, 256 candidates/iteration, dense GPU scoring, CPU candidate generation, and `runtime.use_pmap=false`. Non-random strategies set `random_candidate_fraction=0.0`; `random_mutation` sets it to `1.0`. Existing source-shortfall fallback may still fill unproduced candidate slots with random mutations. Exact true-query metrics are offline evaluation only.

Advantage-gated runs:

| Variant | Final measured loss | True MAE | True RMSE | Accepted edits | Positive candidate rate |
|---|---:|---:|---:|---:|---:|
| `random_mutation` | 65.748090 | 0.00245679 | 0.00389629 | 1747 | 0.039297 |
| `masked_single_query` | 66.800469 | 0.00251852 | 0.00400000 | 1486 | 0.056953 |
| `random_source_directed_exit` | 66.926299 | 0.00251440 | 0.00388253 | 1804 | 0.039734 |
| `single_query` | 68.296449 | 0.00252675 | 0.00403687 | 1512 | 0.055961 |
| `masked_paired_query` | 71.171466 | 0.00252675 | 0.00413258 | 1417 | 0.052691 |
| `paired_query` | 74.945066 | 0.00276132 | 0.00456818 | 1399 | 0.056242 |
| `masked_exit_only` | 75.915075 | 0.00282716 | 0.00451381 | 1496 | 0.049664 |
| `masked_exit_query` | 77.418970 | 0.00267901 | 0.00452565 | 1343 | 0.051754 |
| `directed_exit_only` | 80.424615 | 0.00279424 | 0.00466623 | 1331 | 0.061590 |
| `masked_exit_query_full` | 80.454382 | 0.00287654 | 0.00464058 | 1621 | 0.050695 |
| `masked_paired_query_full` | 87.122663 | 0.00291770 | 0.00534066 | 1368 | 0.050797 |
| `paired_query_full` | 193.221520 | 0.00468724 | 0.00973539 | 630 | 0.049234 |

Blind-accept runs:

| Variant | Final measured loss | True MAE | True RMSE | Accepted edits | Positive diagnostic rate |
|---|---:|---:|---:|---:|---:|
| `blind_paired_query_full` | 278.506187 | 0.00569547 | 0.01080009 | 8000 | 0.672348 |
| `blind_single_query` | 310.742387 | 0.00534156 | 0.00869937 | 8000 | 0.562066 |
| `blind_paired_query` | 335.627908 | 0.00621811 | 0.01134622 | 8000 | 0.607250 |
| `blind_masked_single_query` | 363.255963 | 0.00570782 | 0.00884224 | 8000 | 0.557809 |
| `blind_masked_paired_query` | 827.037922 | 0.00977778 | 0.02146046 | 8000 | 0.630781 |
| `blind_directed_exit_only` | 868.443896 | 0.00934979 | 0.01350476 | 8000 | 0.521574 |
| `blind_masked_paired_query_full` | 1063.841286 | 0.01128807 | 0.02347672 | 8000 | 0.667953 |
| `blind_masked_exit_query` | 1812.981700 | 0.01406584 | 0.02488905 | 8000 | 0.711332 |
| `blind_masked_exit_query_full` | 1906.637216 | 0.01424691 | 0.02483857 | 8000 | 0.537578 |
| `blind_masked_exit_only` | 2583.782594 | 0.01576955 | 0.02466500 | 8000 | 0.529883 |
| `blind_random_source_directed_exit` | 7394.432929 | 0.01875720 | 0.03435508 | 8000 | 0.529586 |
| `blind_random_mutation` | 16246.446272 | 0.03602469 | 0.05849843 | 8000 | 0.468379 |

Selected loss checkpoints:

| Variant | Loss @100 | Loss @500 | Loss @1000 |
|---|---:|---:|---:|
| `random_mutation` | 148.394662 | 66.773442 | 65.748090 |
| `single_query` | 130.518608 | 69.292350 | 68.296449 |
| `masked_single_query` | 126.190833 | 68.258549 | 66.800469 |
| `random_source_directed_exit` | 159.880163 | 69.409682 | 66.926299 |
| `masked_paired_query` | 124.404113 | 74.190536 | 71.171466 |
| `directed_exit_only` | 132.494486 | 80.424615 | 80.424615 |

### Current Judgement

- `masked_single_query` is a small positive measured-loss signal over this run's `single_query` (`66.800` vs `68.296`), but not enough to beat `random_mutation` (`65.748`) or `random_source_directed_exit` (`66.926` on measured loss, slightly better true RMSE).
- The blind runs reinforce that objective gating is necessary. Even the best blind run is far worse than gated QDTE; `blind_random_mutation` is destructive.
- Full-budget paired proposals are risky. `paired_query_full` is clearly bad, and masked/full paired variants underperform minority-budget variants.
- Exit-only variants remain weak or stall, so destination-aware construction or proposal mixture is still the main algorithmic gap.

### Next Recommended Task

- Implement a proposal mixture scheduler rather than choosing one compiler for the whole budget:
  - keep a random reserve;
  - include `masked_single_query` and `single_query` as one-query specialists;
  - include paired/masked paired only as minority destination-aware proposals;
  - avoid full-budget paired/exit-only as defaults.
- Add accepted attribution by `repair_type`, then tune proposal budgets using accepted rate and positive candidate rate.

## Single-Query Mechanism Reminder - 2026-06-10

### What Changed

- No algorithm code changed.
- Added this reminder because `single_query` is easy to confuse with random mutation or paired/destination-aware variants.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a mechanism clarification only.

### Current Status

- `single_query` is the default one-query directed repair compiler.
- Each iteration first selects active measured queries from the DP noisy/projected residuals, using normalized residual-above-noise priority.
- For each active query:
  - if `residual[q] > 0`, the synthetic answer is too low, so candidates are enter edits;
  - if `residual[q] < 0`, the synthetic answer is too high, so candidates are exit edits.
- Enter source rule:
  - sample rows that currently do not satisfy query `q`;
  - repair the row so it satisfies `q`;
  - candidate `repair_type=1`.
- Exit source rule:
  - sample rows that currently satisfy query `q`;
  - repair one breakable query term so the row no longer satisfies `q`;
  - candidate `repair_type=2`.
- It is not destination-aware:
  - an enter edit targets only one underfit query;
  - an exit edit targets only one overfit query;
  - it does not explicitly pair an overfit source with an underfit destination.
- It is still scored globally:
  - candidate generation uses one target query;
  - acceptance uses full-query edit advantage over all measured queries.
- With `random_candidate_fraction > 0`, any leftover/reserved budget adds random mutations; in the cleaned 1000-step ablations most directed variants used `random_candidate_fraction=0.0`, but the default smoke config still has `0.05`.

### Next Recommended Task

- When describing baselines, call `single_query` "one-query directed enter/exit repair", not "random mutation".
- Compare it separately from:
  - `random_mutation`: random proposal plus objective gate;
  - `paired_query` / `masked_paired_query`: destination-aware source/destination repair;
  - `directed_exit_only`: negative-residual exit-only specialization;
  - Private-GSD-style population search, if implemented later.

## Blind-Accept Ablation And Private-GSD Positioning - 2026-06-10

### What Changed

- Added experimental `qdte.transport_mode: blind_accept`.
  - It accepts generated nonconflicting candidates in generation order.
  - It intentionally bypasses candidate positive-advantage filtering and batch/prefix advantage acceptance.
  - It still computes diagnostic `batch_advantage`, but does not use it to accept or reject edits.
- Added `blind_` variant prefix support to `scripts/run_ablation.py`, for example:
  - `blind_random_mutation`
  - `blind_single_query`
  - `blind_directed_exit_only`
- Added tests proving blind transport can accept a negative-advantage batch.
- Clarified the Private-GSD comparison:
  - Private-GSD uses population-level random mutation/crossover and whole-dataset fitness selection over elite datasets.
  - QDTE's stronger distinction is single-dataset, row-edit-level directed evolution: residuals compile into local enter/exit proposals and exact local measured-loss advantage.
  - Therefore "no edit advantage" is not an equivalent Private-GSD baseline. The closer local baseline is random proposal plus the same whole-objective/advantage selection.

### Changed Files

- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_transport.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_transport.py tests/test_config_validation.py`
  - Result: `29 passed in 1.02s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `84 passed in 6.85s`
- `git diff --check`
  - Result: passed

### 1000-Step Blind-Accept Ablation

All blind runs used `configs/smoke.yaml`, seed 0, DP mode, 243 measured queries, 1000 rows, `qdte.max_iters=1000`, `qdte.stop_patience=1000`, 256 candidates/iteration, dense GPU scoring, CPU candidate generation, and `transport_mode=blind_accept`. Exact true-query metrics are offline evaluation only.

| Variant | Advantage-gated loss | Blind-accept loss | Gated true MAE | Blind true MAE | Gated accepted | Blind accepted |
|---|---:|---:|---:|---:|---:|---:|
| `random_mutation` | 65.748090 | 16246.446272 | 0.00245679 | 0.03602469 | 1747 | 8000 |
| `single_query` | 68.296449 | 326.214212 | 0.00252675 | 0.00586008 | 1512 | 8000 |
| `random_source_directed_exit` | 66.926299 | 7394.432929 | 0.00251440 | 0.01875720 | 1804 | 8000 |
| `masked_paired_query` 25% | 71.171466 | 827.037922 | 0.00252675 | 0.00977778 | 1417 | 8000 |
| `masked_exit_only` | 75.915075 | 2583.782594 | 0.00282716 | 0.01576955 | 1496 | 8000 |
| `masked_exit_query` 25% | 77.418970 | 1812.981700 | 0.00267901 | 0.01406584 | 1343 | 8000 |
| `directed_exit_only` | 80.424615 | 868.443896 | 0.00279424 | 0.00934979 | 1331 | 8000 |

### Current Status

- Blind accepting random mutations is destructive on this smoke setup: measured loss rises from the initial `4306.659` to `16246.446`.
- Blind accepting directed candidates is less destructive because the proposal generator has residual signal. For example, `blind_directed_exit_only` reaches `868.444`, much better than blind random mutation.
- However every blind variant is far worse than its advantage-gated counterpart. This confirms that the local measured-loss advantage is not just an implementation detail; it is the acceptance objective that prevents collateral damage from dominating.
- `directed_exit_only` being worst among gated variants is therefore not because advantage scoring is harming a good exit-only process. With the objective gate removed, exit-only accepts many edits and still settles far above gated QDTE. The remaining issue is proposal support/destination construction, not the existence of advantage scoring.

### Current Judgement

- Paper positioning should emphasize:
  - Private-GSD: dataset-population genetic search with random sparse mutations/crossover and whole-dataset fitness selection.
  - QDTE: single synthetic dataset evolved through many local row edits, where noisy/projected query residuals create a directional field and edit advantage gives the exact measured-loss reduction for each local move.
- The main innovation is not "we score candidates and Private-GSD does not"; Private-GSD also uses whole-dataset fitness selection.
- The defensible innovation is "we avoid population-level random search by compiling residual signs into directed row edits and transporting mass within one synthetic table."

### Next Recommended Task

- Add a true Private-GSD-style baseline if needed:
  - maintain a small elite population of synthetic tables;
  - generate random mutation/crossover candidate tables from the current best/elite;
  - score each table by the same DP measured objective;
  - select the top elite tables.
- For QDTE improvement, prioritize destination-aware proposals:
  - keep random proposal reserve;
  - add paired enter/exit destination construction;
  - record accepted attribution by `repair_type`;
  - consider an adaptive proposal mixture scheduler.

## Exit-Axis Ablations And 1000-Step Comparison - 2026-06-10

### What Changed

- Added CPU-only candidate compiler ablations to isolate directed exit design choices:
  - `qdte.candidate_compiler: directed_exit_only`
  - `qdte.candidate_compiler: masked_exit_only`
  - `qdte.candidate_compiler: random_source_directed_exit`
- Added diagnostics for:
  - `directed_exit_only_candidates`
  - `masked_exit_only_candidates`
  - `random_source_directed_exit_candidates`
  - `random_source_exit_attempts`
- Fixed the `random_mutation` ablation definition in `scripts/run_ablation.py`: it now keeps active query selection enabled and only sets `random_candidate_fraction=1.0`. The previous version also set `num_active_targets=0`, causing immediate stop and an invalid random baseline.
- Added unit tests and config-validation coverage for the new compilers.
- Updated README and architecture docs with the new variants and current interpretation.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `36 passed in 0.07s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `82 passed in 6.84s`

### 1000-Step Smoke Ablations

All runs used `configs/smoke.yaml`, seed 0, DP mode, 243 measured queries, 1000 rows, `qdte.max_iters=1000`, `qdte.stop_patience=1000`, 256 candidates/iteration, dense GPU scoring, CPU candidate generation, and microbatch greedy transport. Non-random variants set `random_candidate_fraction=0.0`; `single_query` still produced 7,551 fallback random candidates because source filtering could not always fill the full directed budget. Exact true-query metrics below are offline evaluation only.

| Variant | Final measured loss | True MAE | True RMSE | Accepted edits | Positive candidate rate |
|---|---:|---:|---:|---:|---:|
| `random_mutation` | 65.748090 | 0.00245679 | 0.00389629 | 1747 | 0.039297 |
| `single_query` | 68.296449 | 0.00252675 | 0.00403687 | 1512 | 0.055961 |
| `random_source_directed_exit` | 66.926299 | 0.00251440 | 0.00388253 | 1804 | 0.039734 |
| `masked_paired_query` 25% | 71.171466 | 0.00252675 | 0.00413258 | 1417 | 0.052691 |
| `masked_exit_only` | 75.915075 | 0.00282716 | 0.00451381 | 1496 | 0.049664 |
| `masked_exit_query` 25% | 77.418970 | 0.00267901 | 0.00452565 | 1343 | 0.051754 |
| `directed_exit_only` | 80.424615 | 0.00279424 | 0.00466623 | 1331 | 0.061590 |

Loss checkpoints:

| Variant | Loss @100 | Loss @500 | Loss @1000 |
|---|---:|---:|---:|
| `random_mutation` | 148.394662 | 66.773442 | 65.748090 |
| `single_query` | 130.518608 | 69.292350 | 68.296449 |
| `random_source_directed_exit` | 159.880163 | 69.409682 | 66.926299 |
| `masked_paired_query` 25% | 124.404113 | 74.190536 | 71.171466 |
| `masked_exit_only` | 137.122621 | 78.102562 | 75.915075 |
| `masked_exit_query` 25% | 127.430525 | 77.418970 | 77.418970 |
| `directed_exit_only` | 132.494486 | 80.424615 | 80.424615 |

### Current Status

- The 100-step story and the 1000-step story differ:
  - directed `single_query` and exit-only variants reduce loss faster early;
  - pure random mutation overtakes by 1000 steps on this smoke setup.
- This does not mean the QDTE objective or directed advantage formula is wrong. All variants used the same measured-loss scoring and transport acceptance. The difference is candidate support:
  - directed proposals concentrate on active residual predicates and quickly harvest obvious high-advantage edits;
  - late in the run, the current directed proposal families often return zero positive candidates;
  - random mutation has worse early direction but broader support, so it keeps finding small late improvements.
- `directed_exit_only` is especially weak long-run. It removes positive-residual enter moves and cannot deliberately fill underfit regions.
- `masked_exit_only` helps relative to full `directed_exit_only`, but the exit-only family still underperforms random and single-query.
- `random_source_directed_exit` is better than full source-filtered exit-only and close to random on RMSE, suggesting that overly strict source filtering is a major source of late-stage stagnation.
- Removing the inherited 5% random reserve from `masked_exit_query` and `masked_paired_query` made their 1000-step results worse, which strengthens the case for an explicit exploration budget rather than a purely directed proposal pool.
- The previous 100-step masked/paired results should be treated as early-convergence evidence, not final-quality evidence.

### Current Judgement

- The main innovation should not be stated as "directed is always better than random." The more defensible claim is:
  - QDTE compiles residual direction into structured edit proposals that improve early convergence and positive-candidate density;
  - the current proposal support is too narrow late in optimization;
  - a strong algorithm likely needs an explicit exploration/diversity component instead of replacing random mutation entirely.
- The Private-GSD comparison should therefore separate:
  - random mutation baseline;
  - directed enter/exit compiler;
  - directed source selection;
  - directed edit construction;
  - mixed/random-reserve proposal budgets.

### Next Recommended Task

- Implement an explicit proposal mixture scheduler instead of a single `candidate_compiler`:
  - reserve a nonzero random mutation budget throughout the run;
  - include directed single-query enter/exit for early convergence;
  - include masked exit/paired only as minority specialists;
  - adapt proposal budgets using accepted-rate or positive-candidate-rate by `repair_type`.
- Add per-`repair_type` accepted attribution to runtime/timeseries so the scheduler can learn which proposal family is still productive.
- Re-run multi-seed smoke first, then Adult, because this is currently one seed and one small workload.

## Directed Exit Versus Random Mutation Discussion - 2026-06-10

### What Changed

- Clarified that the recent smoke comparisons did not compare against a Private-GSD-style fully random mutation baseline.
- No implementation code was changed.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm/ablation-design discussion.

### Current Status

- The current `single_query` baseline is already directed:
  - positive residual queries generate directed enter edits;
  - negative residual queries generate directed exit edits;
  - random candidates are only the configured fallback fraction, and recent smoke experiments set `random_candidate_fraction=0.0`.
- `masked_exit_query` is not "random candidate + directed exit." It is a paired/protected proposal:
  - choose a negative-residual source mask;
  - choose a positive-residual destination/protection mask;
  - sample rows satisfying both;
  - edit only source-mask terms to exit while preserving the protected destination mask.
- Therefore masked-exit did not lose to random mutation. It was compared against an already directed one-query compiler and against masked-paired enter.

### Current Judgement

- Private-GSD-style mutation and current QDTE differ along at least two axes:
  - source/old-row choice: random versus residual/query-directed;
  - new-row/edit construction: random mutation versus predicate-compiled enter/exit repair.
- The observed masked-exit weakness at high fractions is likely due to over-constrained proposal support, not because random mutation is inherently better:
  - protected destination masks restrict eligible source rows;
  - source-mask exit without destination enter can create small local moves that help RMSE but not always measured loss/MAE;
  - replacing too much one-query repair reduces proposal diversity.

### Next Recommended Task

- Add explicit ablations that isolate the axes:
  - `random_mutation`: random source and random new row, current ablation exists;
  - `directed_exit_only`: directed negative-residual source and directed exit, no destination protection;
  - `random_source_directed_exit`: random source pool filtered/edited only by negative residual exit;
  - `directed_enter_only`: positive-residual directed enter;
  - current `single_query`, `masked_paired_query`, and `masked_exit_query`.
- Report positive candidate rate, accepted rate, measured loss vs wall-clock, MAE, and RMSE for each.

## Masked Exit-Protect Compiler Implementation And Comparison - 2026-06-10

### What Changed

- Implemented optional CPU `qdte.candidate_compiler: masked_exit_query`.
- This compiler tests the "preserve underfit attributes, exit overfit attributes" idea:
  - selects negative-residual active queries as source regions;
  - selects positive-residual active queries as protected destination regions;
  - samples masks from source and destination ordinary query terms;
  - samples rows satisfying both the source mask and protected destination mask;
  - edits only source-mask terms to exit the source mask;
  - keeps a candidate only if the protected destination mask is still satisfied after the exit edit;
  - leaves final acceptance to the unchanged full measured objective.
- Added `repair_type=5` and diagnostics for `masked_exit_candidates`.
- Added ablation runner variants:
  - `masked_exit_query`: `paired_candidate_fraction=0.25`, `mask_min_terms=1`, `mask_max_terms=2`;
  - `masked_exit_query_full`: `paired_candidate_fraction=1.0`, `mask_min_terms=1`, `mask_max_terms=2`.
- Added tests for masked-exit preservation semantics and config validation.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `30 passed in 0.06s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `76 passed in 6.92s`

### Smoke Experiments Run

All runs used `configs/smoke.yaml`, the same generated smoke CSV and seed, DP mode, `runtime.use_pmap=false`, `qdte.random_candidate_fraction=0.0`, 100 iterations, and 256 candidates/iteration.

Reference results from the previous masked comparison:

- `outputs/exp_maskedcmp_single_100`
  - single query: final measured loss `130.519`, true MAE `0.00364198`, true RMSE `0.00718394`
- `outputs/exp_maskedcmp_masked25_100`
  - 25% masked-paired enter: final measured loss `124.404`, true MAE `0.00346502`, true RMSE `0.00709083`

New masked-exit runs:

- `outputs/exp_exitcmp_maskedexit25_100`
  - `candidate_compiler=masked_exit_query`, `paired_candidate_fraction=0.25`
  - final measured loss: `127.431`
  - final true-query MAE: `0.00367078`
  - final true-query RMSE: `0.00689486`
- `outputs/exp_exitcmp_maskedexit50_100`
  - `candidate_compiler=masked_exit_query`, `paired_candidate_fraction=0.5`
  - final measured loss: `150.896`
  - final true-query MAE: `0.00395062`
  - final true-query RMSE: `0.00782525`
- `outputs/exp_exitcmp_maskedexit100_100`
  - `candidate_compiler=masked_exit_query`, `paired_candidate_fraction=1.0`
  - final measured loss: `194.542`
  - final true-query MAE: `0.00459259`
  - final true-query RMSE: `0.00940537`

### Current Status

- The masked-exit idea has a positive signal at 25% mixture:
  - better measured loss and RMSE than the single-query baseline;
  - worse measured loss and MAE than the 25% masked-paired enter compiler;
  - best RMSE among the compared smoke runs listed above.
- As with paired and masked-paired compilers, 50% and 100% masked-exit are worse. The proposal family is useful as a minority mixture, not as a replacement.
- Default remains `single_query`; all paired/masked compilers are CPU-only experimental proposal families.

### Current Judgement

- The user's hypothesis is partly supported: protecting underfit masks while exiting overfit masks gives useful candidates.
- On this smoke run, masked-paired enter is still the stronger measured-loss/MAE move, while masked-exit gives slightly better RMSE. This suggests the two proposal families may be complementary rather than mutually exclusive.
- The next serious experiment should include a mixed proposal scheduler with separate budgets for:
  - single-query repair;
  - masked-paired enter;
  - masked-exit protect.

### Next Recommended Task

- Implement a proposal-mixture budget config rather than overloading one `paired_candidate_fraction`, e.g.:
  - `proposal_mix.single_query`
  - `proposal_mix.masked_paired_enter`
  - `proposal_mix.masked_exit`
- Add accepted-rate attribution by `repair_type` so the system can adapt proposal budgets online.
- Run multi-seed smoke and Adult CPU sweeps before considering GPU implementation.

## Masked Paired Compiler Implementation And Smoke Comparison - 2026-06-10

### What Changed

- Implemented optional CPU `qdte.candidate_compiler: masked_paired_query`.
- The masked paired compiler:
  - selects negative-residual active queries as source regions and positive-residual active queries as destination regions;
  - samples ordinary query-term masks from source and destination queries;
  - samples rows satisfying the masked source and not satisfying the masked destination;
  - repairs only the masked destination terms, optionally breaking masked source terms;
  - keeps final acceptance on the unchanged full measured objective.
- Added `repair_type=4` and diagnostics for `masked_paired_candidates`.
- Added ablation runner variants:
  - `masked_paired_query`: `paired_candidate_fraction=0.25`, `mask_min_terms=1`, `mask_max_terms=2`;
  - `masked_paired_query_full`: `paired_candidate_fraction=1.0`, `mask_min_terms=1`, `mask_max_terms=2`.
- Added tests for masked-paired partial destination repair and config validation.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `28 passed in 0.06s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `74 passed in 6.95s`

### Smoke Experiments Run

All runs used `configs/smoke.yaml`, the same generated smoke CSV and seed, DP mode, `runtime.use_pmap=false`, `qdte.random_candidate_fraction=0.0`, 100 iterations, and 256 candidates/iteration.

- `outputs/exp_maskedcmp_single_100`
  - `candidate_compiler=single_query`
  - final measured loss: `130.519`
  - final true-query MAE: `0.00364198`
  - final true-query RMSE: `0.00718394`
- `outputs/exp_maskedcmp_paired25_100`
  - `candidate_compiler=paired_query`, `paired_candidate_fraction=0.25`
  - final measured loss: `129.717`
  - final true-query MAE: `0.00360494`
  - final true-query RMSE: `0.00694126`
- `outputs/exp_maskedcmp_masked25_100`
  - `candidate_compiler=masked_paired_query`, `paired_candidate_fraction=0.25`, `mask_min_terms=1`, `mask_max_terms=2`
  - final measured loss: `124.404`
  - final true-query MAE: `0.00346502`
  - final true-query RMSE: `0.00709083`
- `outputs/exp_maskedcmp_masked50_100`
  - `candidate_compiler=masked_paired_query`, `paired_candidate_fraction=0.5`
  - final measured loss: `134.995`
  - final true-query MAE: `0.00376132`
  - final true-query RMSE: `0.00719854`
- `outputs/exp_maskedcmp_masked100_100`
  - `candidate_compiler=masked_paired_query`, `paired_candidate_fraction=1.0`
  - final measured loss: `178.239`
  - final true-query MAE: `0.0045679`
  - final true-query RMSE: `0.00951326`

### Current Status

- Masked paired has a positive signal in the 25% hybrid setting on this smoke run:
  - better measured loss than both single-query and 25% full-paired;
  - best final true-query MAE among the compared runs;
  - true-query RMSE is slightly worse than 25% full-paired but better than single-query.
- Full masked paired remains too narrow as a complete replacement. It does not collapse as badly as full unmasked paired, but it is still worse than the hybrid settings.
- Default remains `single_query`; masked paired is an experimental CPU-only proposal family.

### Current Judgement

- The mask idea appears to address the proposal-collapse problem in the intended direction: it widens search while preserving residual-guided transport structure.
- The right shape is still a mixture, not replacement. Current best smoke result is `25% masked paired + 75% single-query`.
- This strengthens the paper narrative: residuals define transport pressure, but high-order query predicates should be compiled through partial/masked constraints to avoid over-constraining local edits.

### Next Recommended Task

- Run multi-seed and Adult-scale sweeps for `masked_paired_query` over:
  - `paired_candidate_fraction in {0.1, 0.25, 0.5}`;
  - `mask_max_terms in {1, 2, 3}`;
  - `paired_try_break_source in {true, false}`.
- Add accepted-rate attribution by `repair_type` so masked proposals can be compared by accepted positive yield, not only generated count.
- If the signal holds, implement a fixed-shape GPU version as a mixed proposal family.

## Masked Paired Compiler Idea - 2026-06-10

### What Changed

- Analyzed why full paired-query compiler can stall and discussed a masked-query proposal family.
- No implementation code was changed in this discussion.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm-design discussion.

### Historical Status

- The full paired compiler generated candidates late in the smoke run, but all returned advantages became non-positive. This suggests proposal collapse rather than source-filter collapse.
- Likely cause: full active queries are conjunctions over multiple attributes, and one query residual sign can conflate heterogeneous attribute-level errors. A high-order query may be overfit because of one subset of terms while another term or marginal direction is still underfit.
- Full paired repair enforces the complete destination query and may try to break the complete source query. That creates narrow and sometimes high-cost moves, especially when source/destination conjunctions overlap or conflict.
- This idea has since been implemented as `qdte.candidate_compiler: masked_paired_query`; see the top handoff section for current results. The design was:
  - source mask: sample rows satisfying only the overfit-relevant terms;
  - destination mask: repair only selected underfit-relevant terms;
  - leave unmasked attributes unchanged or lightly mutated for diversity;
  - still score and accept with the unchanged full measured objective.

### Current Judgement

- Masking is a plausible way to preserve the paper insight while widening the search space:
  - full paired compiler: strong but narrow `q_minus -> q_plus`;
  - one-query compiler: broad but less explicitly transport-like;
  - masked paired compiler: partial transport guided by residual field, with more feasible nearby `x_new`.
- The DP boundary remains clean if mask selection uses only noisy/projected residuals, variances, query definitions, and synthetic data. Exact true answers must not be used to choose masks.

### Historical Next Recommended Task

- Prototype `candidate_compiler: masked_paired_query` as a CPU-only ablation:
  - sample term masks of size 1 to `max_mask_terms`;
  - prefer masks whose subqueries exist in the measured catalogue and have matching residual sign;
  - otherwise use masked terms only as proposal constraints, not as an optimization target;
  - record mask size, paired/masked candidate counts, positive rate, and accepted rate.
- This prototype and first smoke comparison are now complete; continue with the multi-seed sweeps listed in the top section.

## Paired Query Compiler Implementation And Smoke Experiment - 2026-06-10

### What Changed

- Implemented an optional CPU paired-query candidate compiler behind `qdte.candidate_compiler: paired_query`.
- The paired compiler:
  - selects negative-residual active queries as source regions and positive-residual active queries as destination regions;
  - samples rows that satisfy the source query and do not satisfy the destination query;
  - repairs the row into the destination query;
  - optionally tries to break the source query while preserving destination membership;
  - leaves final acceptance to the unchanged full measured objective score.
- Kept default behavior unchanged: `qdte.candidate_compiler` defaults to `single_query`.
- Added ablation runner variants:
  - `paired_query`: `paired_candidate_fraction=0.25`;
  - `paired_query_full`: `paired_candidate_fraction=1.0`.
- Added a homogeneous-table paired compiler test: all rows start in an overfit source query and are compiled into an underfit destination query.
- Updated README and architecture docs with the new optional compiler and current test status.

### Changed Files

- `qdte/evolution/candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `scripts/run_ablation.py`
- `tests/test_repairs.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_config_validation.py`
  - Result: `25 passed in 0.06s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `71 passed in 6.96s`

### Smoke Experiments Run

All runs used `configs/smoke.yaml`, the same generated smoke CSV, same seed, DP mode, `runtime.use_pmap=false`, `qdte.random_candidate_fraction=0.0`, and 256 candidates/iteration.

- `outputs/exp_compiler_single_50`
  - `candidate_compiler=single_query`, 50 iterations
  - final measured loss: `370.754`
  - final true-query MAE: `0.00626337`
  - final true-query RMSE: `0.0139688`
- `outputs/exp_compiler_paired_50`
  - `candidate_compiler=paired_query`, `paired_candidate_fraction=1.0`, 50 iterations
  - final measured loss: `363.447`
  - final true-query MAE: `0.00652675`
  - final true-query RMSE: `0.0134614`
- `outputs/exp_compiler_single_100`
  - `candidate_compiler=single_query`, 100 iterations
  - final measured loss: `130.519`
  - final true-query MAE: `0.00364198`
  - final true-query RMSE: `0.00718394`
- `outputs/exp_compiler_paired_100`
  - `candidate_compiler=paired_query`, `paired_candidate_fraction=1.0`, `paired_try_break_source=true`, 100 iterations
  - final measured loss: `193.222`
  - final true-query MAE: `0.00468724`
  - final true-query RMSE: `0.00973539`
  - stalled late with positive candidate rate dropping to zero.
- `outputs/exp_compiler_paired50_100`
  - `candidate_compiler=paired_query`, `paired_candidate_fraction=0.5`, 100 iterations
  - final measured loss: `135.776`
  - final true-query MAE: `0.00384774`
  - final true-query RMSE: `0.00717649`
- `outputs/exp_compiler_paired_nobreak_100`
  - `candidate_compiler=paired_query`, `paired_candidate_fraction=1.0`, `paired_try_break_source=false`, 100 iterations
  - final measured loss: `251.359`
  - final true-query MAE: `0.00526337`
  - final true-query RMSE: `0.0112767`
- `outputs/exp_compiler_paired25_100`
  - `candidate_compiler=paired_query`, `paired_candidate_fraction=0.25`, 100 iterations
  - final measured loss: `129.717`
  - final true-query MAE: `0.00360494`
  - final true-query RMSE: `0.00694126`

### Current Status

- The paired compiler idea is implemented and test-covered on the CPU path.
- The current smoke evidence is mixed:
  - full paired proposals improve early loss reduction but become too narrow and can stall;
  - a 25% paired / 75% single-query hybrid slightly beats the single-query baseline on this smoke run;
  - full paired should not replace the default yet.
- The current default remains `single_query` because the evidence is not broad enough for a default change.
- Candidate generation is not pure random: default CPU generation is one-query directed repair plus a configurable random fallback fraction. The new paired compiler is an additional, more structured proposal source.

### Current Judgement

- The paper-level insight is valid: a useful edit is a local transport from negative-residual query regions to positive-residual query regions under the weighted residual field.
- The implementation result suggests the compiler should be hybrid/adaptive, not all-or-nothing. Paired proposals are high-signal early, but need diversity from ordinary one-query repair to avoid late-stage proposal collapse.
- The right next algorithm direction is likely an adaptive mixture scheduler for proposal families, using online positive-rate or accepted-advantage diagnostics.

### Next Recommended Task

- Add experiment-runner support for sweeping `paired_candidate_fraction` over `{0.0, 0.1, 0.25, 0.5, 1.0}` across multiple seeds and reporting loss-vs-wall-clock.
- Add runtime aggregation for `paired_candidates`, paired positive rate, and paired accepted rate; current diagnostics record paired generation counts in timeseries, but accepted-rate attribution is not separated by proposal family.
- If paired remains useful, port a fixed-shape version to the GPU candidate backend as a proposal family mixed with existing one-query directed repair.

## Exit-To-Enter Edit Target Semantics - 2026-06-10

### What Changed

- Clarified the deeper algorithm question: when a source row should exit an overfit query region, what determines the enter region / new row it should become?
- No implementation code was changed.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm-definition discussion.

### Historical Status

- At the time of this discussion, the implementation generated candidates one target query at a time:
  - for an underfit active query, it chooses source rows not satisfying that query and compiles them into rows satisfying it;
  - for an overfit active query, it chooses source rows satisfying that query and compiles them out of it;
  - unchanged attributes are copied from the source row, and edit cost penalizes larger changes;
  - the full measured objective then decides whether the candidate is useful through all-query collateral scoring.
- That implementation did not explicitly solve a paired "exit from q_minus and enter q_plus" assignment. The pairing was implicit: many one-query-directed candidates were generated, and the global edit-advantage score preferred edits whose full delta vector moved mass from overrepresented query features to underrepresented query features. A first optional CPU paired compiler has since been implemented; see the top handoff section.
- The algorithmic insight should be stated more generally: a row edit is a local transport of one synthetic row in query-feature space. The desired new row is determined by the weighted residual field:

```text
w[q] = residual[q] * inv_variance[q]
score(x_old -> x_new) =
  (phi(x_new) - phi(x_old)) @ w
  - 0.5 * ((phi(x_new) - phi(x_old))^2 @ inv_variance)
  - lambda_cost * edit_cost
```

- Under this view, good edits remove mass from regions with negative weighted residual and add mass to regions with positive weighted residual. Query-directed repair supplies feasible high-probability proposals for `x_new`; objective scoring chooses which proposed transports are actually accepted.

### Current Judgement

- This is likely part of the core paper claim:
  - Private-GSD-style random mutation proposes `x_new` without knowing which query region needs mass;
  - QDTE compiles residual signals into feasible membership-changing edits;
  - the full measured objective resolves conflicts between many possible exit/enter effects.
- The code now supports both the original one-query compiler and an optional first paired compiler. Current smoke evidence favors a hybrid mixture rather than full paired replacement.

### Next Recommended Task

- In the algorithm writeup, define the "residual field over query features" and describe candidate repair as a proposal compiler for high-advantage local transports.
- For implementation, continue evaluating the optional paired compiler under equal candidate budgets before making it the default.

## Directed Enter Source Semantics Discussion - 2026-06-10

### What Changed

- Reviewed and clarified how QDTE turns residual signs into directed enter/exit edits.
- No implementation code was changed in this discussion.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm/implementation semantics discussion.

### Current Status

- Current CPU candidate generation separates source selection from edit compilation:
  - if `residual[q] > 0`, QDTE needs an enter edit and samples rows that currently do not satisfy query `q`;
  - if `residual[q] < 0`, QDTE needs an exit edit and samples rows that currently do satisfy query `q`;
  - `_repair_enter_batch` then edits the query terms so the selected old row becomes a new row satisfying `q`;
  - `_repair_exit_batch` edits at least one breakable query term so the selected old row stops satisfying `q`.
- This means enter edits do not require already-satisfying rows to exist. In an extreme homogeneous table where all rows satisfy an overfit query, that query can generate exit edits; another underfit query can still generate enter edits only if its source filter can find rows not satisfying that underfit query.
- The key algorithmic idea is therefore query-directed edit compilation: active query residuals compile into local row-edit programs, and the full measured objective scores collateral effects.
- Current GPU candidate generation follows the same sign logic but, when no source draw matches the desired source satisfaction, falls back to the first draw before repair. This keeps GPU shapes fixed but should be called out and tested as an approximate source-selection behavior.

### Current Judgement

- This source-to-edit mechanism is central to the QDTE innovation claim: QDTE is not waiting for random mutations to stumble into a target query cell; it directly edits the attributes that define the active query.
- The method still depends on schema cardinalities and query feasibility. If a query target is impossible under the encoded schema or all relevant attributes have cardinality one, enter/exit repair can degenerate into no-op or random fallback and should be surfaced by diagnostics.
- For the paper/algorithm definition, source selection should be described as choosing old records with the desired current membership, while repair is the compiler that maps them to the desired opposite membership.

### Next Recommended Task

- Add a small source-selection/repair feasibility test for the homogeneous-table edge case:
  - overfit query with all rows satisfying it should yield exit candidates;
  - underfit feasible query with rows not satisfying it should yield enter candidates;
  - infeasible query or unbreakable query should be recorded as repair/source-filter failure rather than silently treated as strong evidence.
- Decide whether GPU fallback-on-no-match should remain as an approximation or be changed to mark those candidates invalid when no desired source row is found.

## Directed Repair Contracts And Multi-Word Sparse GPU Bitsets - 2026-06-10

### What Changed

- Added focused directed repair contract coverage:
  - k-way `EQ/LE/GE/RANGE` conjunction enter/exit repair;
  - halfspace enter/exit repair on the CPU repair path;
  - `generate_candidates` directed outputs for active queries, while ignoring random fallback candidates with target id `-1`.
- Removed the old sparse GPU single-word query-scope bitset limit:
  - added `QueryDeltaIndex.query_attr_words(...)`;
  - kept `query_attr_bits(...)` only as a single-word compatibility helper;
  - changed the fused sparse GPU scorer to use multi-word `uint32` query-scope bitsets for duplicate suppression.
- Added a sparse GPU regression test that crosses the old 31-attribute boundary and verifies the sparse GPU score against the dense delta objective.
- Updated README and architecture docs to reflect the new implementation status, test status, current judgement, and next work plan.

### Changed Files

- `qdte/queries/delta_index.py`
- `qdte/evolution/gpu_candidates.py`
- `tests/test_gpu_candidates.py`
- `tests/test_repairs.py`
- `README.md`
- `architecture.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_repairs.py tests/test_delta_index.py tests/test_gpu_candidates.py`
  - Result: `17 passed in 3.23s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `68 passed in 6.97s`

### Current Status

- Task 1 is done: directed repair now has explicit contract tests for ordinary k-way queries, halfspace queries, and generated directed candidate batches.
- Task 2 is done: `sparse_delta_gpu` no longer has the old 31 encoded-attribute limit from a single `uint32` bitset. The current GPU duplicate-suppression metadata is `(num_queries, ceil(num_attrs / 32))`.
- The QDTE objective and DP boundary were preserved. Sparse GPU scoring still computes the same edit-advantage formula against measured residuals and inverse variances.
- The remaining sparse GPU capacity assumption is `gpu_sparse_changed_attr_capacity`: it must cover how many attributes a candidate edit can change. Current Adult highpower sets this to `4`, matching `workload.max_terms: 4`.

### Current Judgement

- More trustworthy after this task:
  - directed CPU repair semantics for enter/exit edits;
  - sparse GPU duplicate suppression across more than 31 encoded attributes;
  - agreement between sparse GPU scoring and the dense delta objective on the new boundary test.
- Still needs focused attention:
  - GPU fused directed repair still does not support halfspace workloads;
  - `gpu_sparse_changed_attr_capacity` is still a configuration/runtime assumption, not an automatically proven bound;
  - batch atom-flow is an approximate transport selector, while exact atom-flow remains the cleaner algorithmic reference;
  - baseline systems and matched experiment runners are still missing.

### Next Recommended Task

- Let the algorithm-definition writeup formalize the main QDTE claim as residual-guided, query-directed edit compilation rather than random mutation.
- Engineering next: build the baseline/ablation experiment runner for Private-GSD-style random mutation, QDTE directed repair, target-only scoring, exact vs batch atom-flow, and sparse vs dense scoring under matched workloads and privacy budgets.
- If halfspace workloads are central to the experiment, implement GPU directed halfspace repair or keep halfspace experiments on the CPU repair backend and document that scope explicitly.

## QDTE Completion Gap Assessment - 2026-06-10

### What Changed

- Assessed whether the current implementation is already a real QDTE algorithm and what remains to make it publication/experiment ready.
- No implementation code was changed.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was an algorithm/readiness assessment.

### Current Status

- The repository already implements the core QDTE loop: static DP measurement, noisy/projected target residuals, active query selection, directed enter/exit candidate repair, exact edit-advantage scoring, transport, incremental residual update, and recompute drift checks.
- The current implementation should be considered a runnable QDTE prototype, not yet a fully finalized research artifact.
- Main gaps before a strong QDTE claim:
  - formalize the algorithm around residual-guided query-directed edit compilation;
  - strengthen and test repair contracts across all supported query families;
  - decide whether exact atom-flow or batch atom-flow is the canonical transport algorithm;
  - remove GPU halfspace and sparse bitset limitations if highpower mixed workloads are central;
  - build matched Private-GSD/AIM/MST/RAP++ baseline experiments;
  - add sample-efficiency and ablation experiments that isolate directed mutation from transport and GPU engineering.

### Next Recommended Task

- Historical next step: the directed repair contract tests and multi-word sparse GPU bitset change are now complete; the remaining part is the baseline runner plan.

## Innovation And Baseline Discussion - 2026-06-10

### What Changed

- Assessed the current QDTE method's plausible novelty against existing DP tabular synthetic data lines.
- No implementation code was changed.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- No tests were run; this was a research-design/documentation discussion.

### Current Status

- QDTE's plausible contribution is not a new DP measurement mechanism. Current DP mode is static measurement plus post-processing.
- The strongest plausible novelty is the generate/post-processing step:
  - query-directed discrete row edits;
  - exact quadratic edit-advantage scoring against noisy/projected targets;
  - atom-flow transport over old-row/new-row encoded atoms;
  - sparse affected-query GPU scoring for large candidate pools.
- The closest methodological competitor is Private-GSD, because both are zeroth-order/discrete synthetic-data optimizers for statistical query workloads, including non-differentiable query classes.
- The core baselines should include Private-GSD, AIM/MST/Private-PGM, RAP/RAP++, GEM/PEP where practical, plus targeted QDTE ablations.

### Next Recommended Task

- Before claiming novelty, implement/run baseline comparisons under matched privacy budgets, preprocessing, measured workloads, and held-out workloads.
- Prioritize Private-GSD and AIM/MST first; they are the most important external baselines for QDTE's intended claims.

## Documentation Sync - 2026-06-10

### What Changed

- Synchronized `README.md`, `architecture.md`, and `architecture_zh.md` with the current implementation status.
- Updated stale English architecture details for:
  - Adult/highpower config defaults;
  - `config_validation.py`, `delta_index.py`, and `measurement/consistency.py`;
  - halfspace support in measurement/evaluation/CPU repair/consistency projection;
  - sparse-delta GPU scoring and its current constraints;
  - batch/exact atom-flow transport;
  - held-out evaluation outputs;
  - then-current test coverage and `64 passed` status.
- Synchronized the stale test-status and sparse GPU probe numbers in `architecture_zh.md`.
- Updated README test status and documented the then-current `sparse_delta_gpu` 31-attribute bitset limit.

### Changed Files

- `architecture.md`
- `architecture_zh.md`
- `README.md`
- `docs/HANDOFF.md`

### Tests And Checks Run

- `rg -n '13 passed|halfspace workload construction is not implemented|include_halfspace.*compatibility|131072|gpu_return_top_k: 4096|accepted_per_iter: 512|64 passed' README.md architecture.md architecture_zh.md docs/HANDOFF.md`
  - Result: only then-current/historical handoff entries and `64 passed` references remained; stale architecture content was removed.
- No pytest run after this docs-only edit. The immediately preceding repository audit ran `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q` with `64 passed in 23.29s`.

### Current Status

- README, `architecture.md`, and `architecture_zh.md` now describe the same current implementation state at a high level.
- The codebase remains unchanged by this sync.

### Next Recommended Task

- Historical next step, now superseded by the top section: the `sparse_delta_gpu` 31-attribute bitset limit has been removed with multi-word bitsets.

## Current Repository Audit - 2026-06-10

### What Changed

- Reviewed the current code and documentation to assess implementation status, completed features, remaining gaps, and audit-risk areas.
- No implementation code was changed.
- Confirmed the current test suite passes in the `qdte` conda environment.

### Changed Files

- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `64 passed in 23.29s`

### Current Status

- The repository now contains a runnable QDTE synthetic tabular data generator with CSV preprocessing, workload construction, DP/oracle measurement, projection, one-way initialization, edit-based QDTE optimization, CPU and GPU candidate/scoring paths, atom-flow transport, held-out offline evaluation, and audit outputs.
- The core QDTE objective invariants are implemented in the measured target path:
  - `residual = target_projected - answer_syn`
  - `measured_loss = 0.5 * sum residual^2 * inv_variance`
  - candidate scoring uses `delta @ (residual * inv_variance) - 0.5 * delta^2 @ inv_variance - lambda_cost * edit_cost`
- The DP boundary appears preserved in code paths inspected: in DP mode exact real query answers are used for Gaussian measurement and final/held-out offline evaluation only, while active query selection, candidate generation, scoring, transport, stopping, and state updates use noisy/projected targets, residuals, variances, and synthetic data.
- README, `architecture.md`, and `architecture_zh.md` are now aligned at a high level after the documentation sync above.
- Current highpower path is `candidate_backend: jax_repair`, `score_backend: sparse_delta_gpu`, and batch atom-flow. This is supported by tests and recorded probes, but should still be treated as a performance/quality experiment rather than a fully mature production path.

### Trustworthy Areas

- Objective math and sign convention are covered by focused tests.
- DP measurement budget allocation, `rho_spent`, missing allocation failure, and no true-answer leakage in public measurement JSON are covered.
- Workload/query evaluation supports `oneway`, `twoway`, `prefix`, `range`, `mixed`, `kway`, `kway_prefix`, `kway_range`, `kway_mixed`, `orthogonal_kway_mixed`, and `halfspace`.
- Consistency projection supports `EQ/LE/GE/RANGE` and halfspace masks through scope-local marginal tables, with fail-fast `max_scope_cells`.
- Batch/exact atom-flow and sparse delta paths have unit coverage plus smoke/probe history.
- Full test suite passes as of this audit.

### Areas Requiring Attention

- `architecture.md` needs synchronization with `README.md` / `architecture_zh.md`.
- Projection currently keeps diagonal variances after consistency projection; full post-projection covariance propagation is not implemented.
- GPU fused candidate repair does not support halfspace and intentionally fails fast for `include_halfspace: true` with `jax_repair`/`gpu_repair`.
- `sparse_delta_gpu` assumes the candidate changes no more attributes than `gpu_sparse_changed_attr_capacity`; current Adult highpower sets this to `4`, matching `workload.max_terms: 4`.
- Adaptive select-measure-generate, downstream ML evaluation, public schema loading, plausibility materialization, and baseline systems remain unimplemented.
- Held-out evaluation can collapse to a narrow family after duplicate filtering; prior long-run evidence mostly covered held-out 2-way queries, not all query families.
- There is no packaging/dependency manifest in the repo; the documented conda environment is required for reliable tests/runs.

### Next Recommended Task

- Synchronize `architecture.md` with the current README/Chinese architecture document, especially test status, halfspace support, consistency projection, sparse GPU scoring, and batch atom-flow.
- Then run a longer highpower held-out experiment with a held-out workload that retains prefix/range/mixed queries after duplicate filtering.

## GPU Sparse-Delta Scoring

### What Changed

- Added a compiled GPU sparse-delta scoring path under `qdte.score_backend: sparse_delta_gpu`.
- Extended `QueryDeltaIndex` with GPU-friendly metadata:
  - padded `attr -> affected qids` arrays;
  - then-current query attribute scope bitsets for duplicate suppression.
- Extended `GpuCandidateContext` with sparse scoring constants and device arrays.
- Added `_score_rows_sparse_delta` in `qdte/evolution/gpu_candidates.py`.
  - It runs inside the fused JAX `pmap` candidate-generation/scoring path.
  - It iterates over changed attrs and sparse query blocks.
  - At that point it used single-word `uint32` scope bitsets to avoid counting the same query more than once when a k-way query touches multiple changed attrs. This has since been superseded by the multi-word implementation described at the top of this handoff.
  - It supports ordinary k-way `EQ/LE/GE/RANGE` conjunctions represented by `QueryCatalogue`.
- Added config validation for `score_backend: sparse_delta_gpu`; it requires `candidate_backend` to be `jax_repair` or `gpu_repair`.
- Switched `configs/adult_qdte_gpu_highpower.yaml` to `score_backend: sparse_delta_gpu`.
- Added runtime fields:
  - `gpu_sparse_score_mode`
  - `gpu_sparse_query_block_size`
  - `gpu_sparse_query_block_count`
  - `gpu_sparse_changed_attr_capacity`
- Updated README and architecture docs with the new highpower default and benchmark.

### Changed Files

- `qdte/queries/delta_index.py`
- `qdte/evolution/gpu_candidates.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `configs/adult_qdte_gpu_highpower.yaml`
- `tests/test_gpu_candidates.py`
- `tests/test_config_validation.py`
- `README.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests And Probes Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_gpu_candidates.py tests/test_delta_index.py tests/test_config_validation.py`
  - Result: `24 passed in 2.46s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `64 passed in 6.01s`
- Smoke sparse GPU path:
  - `outputs/bench_smoke_sparsegpu_2b`
  - `score_backend=sparse_delta_gpu`
  - Result: completed 2 iterations, `final_measured_loss=3859.97`.
- Adult highpower dense current-code baseline:
  - `outputs/bench_gpu_dense_jaxprefix_3c`
  - 3 iterations, `time_scoring_seconds=9.9535`, `time_generation_seconds=10.9899`, `candidates_scored_per_second=429359`.
- Adult highpower sparse GPU 3-iteration probe:
  - `outputs/bench_gpu_sparsedelta_gpu_3b`
  - 3 iterations, `time_scoring_seconds=6.3290`, `time_generation_seconds=7.3989`, `candidates_scored_per_second=637743`.
- Adult highpower sparse GPU 50-iteration probe:
  - `outputs/bench_gpu_sparsedelta_gpu_50`
  - `num_candidates_scored=78643200`
  - `num_accepted_edits=36617`
  - `final_measured_loss=127132`
  - `time_generation_seconds=19.9059`
  - `time_scoring_seconds=14.2563`
  - `time_transport_seconds=5.5839`
  - `candidate_scoring_throughput_per_second=5516380`
  - `candidates_scored_per_second=3950755`
  - `accepted_edits_per_second=1839.51`
- Dense GPU 50-iteration reference from previous probe:
  - `outputs/bench_gpu_dense_jaxprefix_50`
  - `time_generation_seconds=36.6483`
  - `time_scoring_seconds=30.1464`
  - `candidates_scored_per_second=2145889`
  - `accepted_edits_per_second=982.174`
  - `final_measured_loss=130806`

### Current Status

- GPU sparse-delta scoring is now the highpower default and is materially faster than dense GPU query-block scoring on the current Adult workload.
- The path is still exact for represented ordinary query conjunctions, but it assumes current GPU repair changes at most `gpu_sparse_changed_attr_capacity` attrs per candidate. The highpower config sets this to `4`, matching `workload.max_terms: 4`.
- Halfspace remains unsupported for GPU fused candidate repair and still fails fast with `candidate_backend: jax_repair`/`gpu_repair`.
- Sparse CPU scoring remains opt-in and is still slower than dense JAX for the tested workloads.

### Next Recommended Task

- Run a longer 500- or 1000-iteration highpower comparison with `score_backend: sparse_delta_gpu` and held-out evaluation enabled.
- Tune `gpu_sparse_query_block_size` over `{32, 64, 128}` and `gpu_sparse_changed_attr_capacity` if future workloads use larger k-way queries.
- If halfspace highpower workloads are needed, implement directed halfspace repair in the GPU fused candidate backend before enabling halfspace with sparse GPU scoring.

## Sparse Delta Performance Probe

### What Changed

- No code changes in this probe.
- Benchmarked the new sparse delta backend against the existing dense JAX/GPU delta path.

### Commands Run

- `env CUDA_VISIBLE_DEVICES=0,1 conda run -n qdte python scripts/run_qdte.py --config configs/adult_qdte_gpu_highpower.yaml --run.output_dir outputs/bench_gpu_dense_jaxprefix_50 --qdte.max_iters 50 --qdte.log_every 10 --qdte.eval_every 50 --evaluation.compute_true_query_error false --evaluation.compute_heldout_query_error false --evaluation.save_synthetic_csv false`
- `env CUDA_VISIBLE_DEVICES=0,1 conda run -n qdte python scripts/run_qdte.py --config configs/adult_qdte_gpu_highpower.yaml --run.output_dir outputs/bench_gpu_jaxprefix_3 --qdte.max_iters 3 --qdte.log_every 1 --qdte.eval_every 3 --qdte.full_recompute_every 0 --evaluation.compute_true_query_error false --evaluation.compute_heldout_query_error false --evaluation.save_synthetic_csv false`
- `env CUDA_VISIBLE_DEVICES=0,1 conda run -n qdte python scripts/run_qdte.py --config configs/adult_qdte_gpu_highpower.yaml --run.output_dir outputs/bench_gpu_sparsecpu_3 --qdte.max_iters 3 --qdte.log_every 1 --qdte.eval_every 3 --qdte.transport_delta_backend sparse_cpu --qdte.full_recompute_every 0 --evaluation.compute_true_query_error false --evaluation.compute_heldout_query_error false --evaluation.save_synthetic_csv false`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/bench_smoke_dense_100 --evaluation.compute_true_query_error false --evaluation.save_synthetic_csv false`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_qdte.py --config configs/smoke.yaml --run.output_dir outputs/bench_smoke_sparse_100 --qdte.score_backend sparse_delta --qdte.transport_delta_backend sparse_cpu --evaluation.compute_true_query_error false --evaluation.save_synthetic_csv false`

### Results

- Adult highpower, default `transport_delta_backend: jax_prefix`, 3 iterations:
  - `wall_clock_seconds=13.1391`
  - `time_generation_seconds=10.9018`
  - `time_scoring_seconds=9.8808`
  - `time_transport_seconds=1.0102`
  - `candidates_scored_per_second=432828`
  - `accepted_edits_per_second=281.789`
  - `final_measured_loss=2274880.22`
- Adult highpower, `transport_delta_backend: sparse_cpu`, 3 iterations:
  - `wall_clock_seconds=94.9516`
  - `time_generation_seconds=92.6832`
  - `time_scoring_seconds=9.9637`
  - `time_transport_seconds=82.7089`
  - `candidates_scored_per_second=50911`
  - `accepted_edits_per_second=33.1452`
  - `final_measured_loss=2274880.22`
- Adult highpower, default `jax_prefix`, 50 iterations:
  - `wall_clock_seconds=42.0606`
  - `time_generation_seconds=36.6483`
  - `time_scoring_seconds=30.1464`
  - `time_transport_seconds=6.4537`
  - `num_candidates_scored=78643200`
  - `num_accepted_edits=35995`
  - `candidates_scored_per_second=2145889`
  - `accepted_edits_per_second=982.174`
  - `final_measured_loss=130806`
- Smoke CPU candidate pool, default dense/JAX scoring, 100 iterations:
  - `wall_clock_seconds=0.7857`
  - `time_scoring_seconds=0.2048`
  - `time_transport_seconds=0.1472`
  - `candidates_scored_per_second=53307`
  - `final_measured_loss=124.444`
- Smoke CPU candidate pool, `score_backend: sparse_delta` and `transport_delta_backend: sparse_cpu`, 100 iterations:
  - `wall_clock_seconds=4.4230`
  - `time_scoring_seconds=3.8513`
  - `time_transport_seconds=0.1298`
  - `candidates_scored_per_second=6227`
  - `final_measured_loss=124.444`

### Current Status

- The current sparse delta implementation is correct but not a performance win. It is Python/CPU-side and loses badly to vectorized dense JAX, especially after GPU top-k returns 16k candidates per iteration.
- It should remain opt-in for debugging and as a correctness baseline for future sparse kernels.
- The performance path for highpower runs remains dense GPU scoring plus JAX prefix delta transport.

### Next Recommended Task

- Do not make `sparse_delta` or `sparse_cpu` the default.
- If sparse delta is still desired for speed, implement it as compiled JAX/GPU segmented affected-query evaluation, not Python per-candidate loops.
- A near-term practical optimization is to reduce returned top-k or move atom-flow grouping/prefix work deeper into JAX/GPU before attempting sparse query indexing again.

## Sparse Candidate Delta Index

### What Changed

- Added `qdte/queries/delta_index.py` with `QueryDeltaIndex`.
- The index maps every attribute to all queries whose ordinary terms or halfspace linear terms can be affected by edits to that attribute.
- Sparse candidate delta remains exact:
  - it first narrows each edit to queries sharing at least one changed attribute;
  - then evaluates the original `QueryCatalogue` predicates on the old and new row;
  - it supports arbitrary k-way `EQ/LE/GE/RANGE` conjunctions up to the catalogue `max_terms`;
  - it includes halfspace queries through their linear attrs.
- Added `compute_deltas_sparse` and `score_candidates_sparse`.
- Added optional `qdte.score_backend: sparse_delta` for CPU candidate scoring.
- Added optional `qdte.transport_delta_backend: sparse_cpu` for microbatch transport and atom-flow transport delta computation.
- Kept existing dense JAX/GPU paths unchanged by default. The high-throughput fused GPU scorer still uses dense predicate evaluation unless the sparse CPU backend is explicitly selected.
- Verified the existing consistency projection is already k-way capable: it builds local marginal tables by full query scope and has existing 3-way, 4-way, and halfspace coverage.

### Changed Files

- `qdte/queries/delta_index.py`
- `qdte/evolution/scoring.py`
- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `tests/test_delta_index.py`
- `tests/test_config_validation.py`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q tests/test_delta_index.py tests/test_transport.py tests/test_config_validation.py`
  - Result: `22 passed in 10.79s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `62 passed in 9.51s`

### Current Status

- Sparse delta is implemented as an opt-in exact backend, not as a replacement for the default GPU dense scorer.
- Query order is not restricted to 1way/2way. Any ordinary conjunction represented in `QueryCatalogue` is handled by the same index/evaluate path, including 3way/4way/kway terms and range/prefix predicates.
- Halfspace deltas are supported by the sparse index, because changed linear attrs mark the halfspace query as affected.
- Optimization objective semantics are unchanged: accepted edits still update `answer_syn += delta_sum` and `residual = target - answer_syn`.

### Next Recommended Task

- Benchmark `transport_delta_backend: sparse_cpu` and `score_backend: sparse_delta` on small/medium CPU candidate pools to determine where sparse CPU wins over dense JAX.
- For highpower GPU runs, the larger next step is a real GPU sparse-delta kernel or segmented affected-query kernel; the current opt-in sparse path is exact but CPU-side.

## Scope-Local Measurement Consistency Projection

### What Changed

- Added `qdte/measurement/consistency.py`.
- Implemented `project_consistent_targets` for supported `EQ/LE/GE/RANGE` query conjunctions:
  - maps every query to its attribute scope;
  - builds a local marginal table for each scope;
  - fits noisy query answers with inverse-variance weights;
  - enforces non-negativity and known total row count `N` for every scope table;
  - reconciles overlapping scopes through shared marginal consistency;
  - returns projected query answers and diagnostics.
- Wired consistency projection into `measure_real_dataset` under:

```yaml
projection:
  consistency:
    enabled: true
    method: local_marginal_ipf
```

- Added `projection_diagnostics` to `measurements.json`.
- Passed `schema.cardinalities` into measurement from the engine, so consistency projection uses authoritative domain sizes rather than inferring them from partial query coverage.
- Enabled `projection.prefix_monotonicity: true` and `projection.consistency.enabled: true` in both Adult configs.
- Added config validation for consistency projection settings.
- Added focused coverage for `GE`, four-dimensional conjunctions, and `max_scope_cells` fail-fast behavior.
- Added halfspace as a first-class query family for measurement/evaluation/CPU repair/consistency projection:
  - `QueryCatalogue` now stores linear halfspace attrs/weights/thresholds;
  - JAX and NumPy query evaluation handle halfspace predicates;
  - workload construction supports `include_halfspace` and `halfspace_queries`;
  - CPU repair can enter/exit halfspace queries;
  - consistency projection maps halfspace masks to local scope marginal cells.
- At that time, added fail-fast validation for `include_halfspace: true` with `candidate_backend` set to `jax_repair`/`gpu_repair`, because GPU fused repair did not yet implement directed halfspace repair. This historical limitation is superseded by the 2026-06-17 GPU fused halfspace repair update at the top of this file.
- Documented the new projection in README and architecture docs.

### Changed Files

- `qdte/measurement/consistency.py`
- `qdte/measurement/measure.py`
- `qdte/evolution/engine.py`
- `qdte/evolution/candidates.py`
- `qdte/config_validation.py`
- `qdte/queries/types.py`
- `qdte/queries/eval_jax.py`
- `qdte/queries/workload.py`
- `configs/adult_qdte.yaml`
- `configs/adult_qdte_gpu_highpower.yaml`
- `tests/test_queries.py`
- `tests/test_workload.py`
- `tests/test_consistency_projection.py`
- `tests/test_measurement.py`
- `tests/test_engine_smoke.py`
- `README.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests And Probes Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_consistency_projection.py -q`
  - Result: `5 passed in 0.21s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_queries.py tests/test_workload.py tests/test_consistency_projection.py tests/test_config_validation.py tests/test_engine_smoke.py::test_engine_halfspace_cpu_smoke_with_consistency_projection -q`
  - Result: `24 passed in 1.50s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_consistency_projection.py tests/test_config_validation.py -q`
  - Result: `17 passed in 0.23s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_consistency_projection.py tests/test_projection.py tests/test_measurement.py tests/test_engine_smoke.py tests/test_config_validation.py -q`
  - Result: `27 passed in 2.83s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `57 passed in 4.95s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/smoke_qdte.py`
  - Result: passed.
- `configs/adult_qdte.yaml` with `--qdte.max_iters 1 --evaluation.compute_true_query_error false --evaluation.save_synthetic_csv false`
  - Result: passed.
  - `time_measurement_seconds`: `1.5159345799984294`.
  - Consistency diagnostics: `num_scopes=82`, `num_scope_cells=11298`, `max_scope_cells_observed=656`, `iterations=14`, `max_marginal_error=0.006001949310302734`, `tolerance=0.01`, `converged=true`, `known_total_count=45222`.

### Current Status

- Consistency projection is implemented for all currently supported query algebra operations: `EQ`, `LE`, `GE`, and `RANGE`.
- This covers the implemented workload families: `oneway`, `twoway`, `prefix`, `range`, `mixed`, and `halfspace`, plus future adaptive supported conjunctions of arbitrary dimension when the resulting scope table is within `max_scope_cells`.
- Halfspace is implemented for measurement/evaluation/CPU repair/consistency projection. GPU fused candidate repair for halfspace is still unsupported and explicitly fails fast.
- Projection uses the known row count as a hard total-count constraint for local marginal tables.
- The projected covariance is still represented by the existing diagonal variances; full post-projection covariance propagation remains future work.
- Completion audit: the user-requested six-family consistency projection is now implemented for all six workload families in the CPU measurement/evaluation/repair path. The remaining unsupported piece is highpower GPU fused candidate repair for halfspace, which is outside the consistency projection layer and fails fast.

### Next Recommended Task

- Run a longer Adult comparison with consistency projection enabled vs disabled, using held-out workloads that actually cover prefix/range/mixed after duplicate filtering.
- If consistency improves measured and held-out quality, use it as the required measurement layer before implementing adaptive query selection.
- Implement directed halfspace repair in the GPU fused candidate backend if highpower halfspace workloads are needed.

## 1000-Iteration Batch Atom-Flow Held-Out Run

### What Changed

- No code changes were made for this run.
- Ran the current highpower batch atom-flow configuration for 1000 iterations with held-out query evaluation enabled.
- Output directory: `outputs/adult_qdte_gpu_batch_1000_heldout`.

### Command Run

- `env CUDA_VISIBLE_DEVICES=0,1 conda run -n qdte python scripts/run_qdte.py --config configs/adult_qdte_gpu_highpower.yaml --run.output_dir outputs/adult_qdte_gpu_batch_1000_heldout --qdte.max_iters 1000 --qdte.log_every 50 --evaluation.compute_heldout_query_error true --evaluation.heldout_exclude_measured_queries true`

### Results

- Run completed on two CUDA devices.
- Iterations: `1000`.
- Candidates scored: `1572864000`.
- Accepted edits: `63640`.
- Final incremental answer drift: `0`.
- Initial measured loss: `2908450.0216422295`.
- Final measured loss: `3752.127586477457`.
- Runtime generation seconds: `503.74876390100326`.
- Runtime scoring seconds: `420.62622242897487`.
- Runtime transport seconds: `82.59349047602882`.
- Candidates scored per second: `3122318.3315028427`.
- Accepted edits per second: `126.33281619824785`.
- Measured-workload true MAE: `0.001720775497623961 -> 0.00029885719769823336`.
- Measured-workload true RMSE: `0.0074053211381847955 -> 0.0011656374406343627`.
- Held-out queries: `6571`.
- Held-out true MAE: `0.0011362534928970544 -> 0.0007739023677580094`.
- Held-out true RMSE: `0.006420617097978189 -> 0.005143872391296755`.

### Issues Found

- No runtime failure, residual drift, or full-recompute mismatch was observed.
- Accepted edits per logged iteration dropped sharply over time: e.g. `1024` at iteration 1, `62` at iteration 100, and `3` at iteration 1000. This suggests late-stage candidate generation/scoring is still expensive relative to accepted improvements.
- Held-out evaluation was positive in this run, so the previous short-run true-query fluctuation did not persist here.
- The held-out workload was only `twoway` after duplicate filtering: `10000` held-out 2-way candidates were constructed and `3429` measured duplicates were removed, leaving `6571`. This run does not prove generalization for held-out prefix/range/mixed families.

### Current Status

- Long-run batch atom-flow is stable on the Adult highpower config.
- The run gives encouraging evidence that the current batch atom-flow does not simply overfit noisy measured queries in the first 1000 iterations for held-out 2-way queries.
- The main remaining algorithmic gap is still full GPU atom-flow / stronger late-stage transport: the current batch approximation is fast and stable but late iterations spend 1.57M scored candidates to accept only a handful of edits.

### Next Recommended Task

- Implement adaptive late-stage candidate allocation or a fuller GPU atom-flow/matching-style selector so late iterations do not keep scoring 1.57M candidates when only a few edits survive the batch objective.
- Expand held-out workload construction so duplicate filtering does not leave only 2-way held-out queries, then rerun the same long evaluation across prefix/range/mixed families.

## Audit-Critical QDTE Fixes

### What Changed

- Added `qdte/config_validation.py` and fail-fast validation for unsupported measurement modes, halfspace workload, downstream ML, unsupported init methods, and unknown backend strings.
- Tightened DP measurement budget allocation so explicit allocations must cover every observed workload family and spend no more than `rho_total`.
- Added `rho_spent` to `Measurements`, `measurements.json`, and `metrics_final.json`.
- Made active query selection strict by default: queries below `kappa_noise * sigma` no longer trigger fallback candidate generation unless `allow_below_noise_fallback` is true.
- Added real query debt updates based on per-query weighted loss damage/improvement after accepted edits.
- Added debt diagnostics to `metrics_timeseries.csv` and `runtime.json`.
- Added weighted PAVA prefix monotonicity projection for prefix measurement groups.
- Made adult configs explicit about `allow_below_noise_fallback: false` and debt scheduler parameters.
- Updated README at that point to say current transport was prefix-greedy microbatch transport; this is superseded by the later Atom-Flow Transport section below.

### Changed Files

- `qdte/config_validation.py`
- `qdte/evolution/engine.py`
- `qdte/evolution/scheduler.py`
- `qdte/measurement/measure.py`
- `qdte/measurement/projection.py`
- `configs/adult_qdte.yaml`
- `configs/adult_qdte_gpu_highpower.yaml`
- `tests/test_config_validation.py`
- `tests/test_engine_smoke.py`
- `tests/test_measurement.py`
- `tests/test_projection.py`
- `tests/test_scheduler.py`
- `README.md`
- `docs/HANDOFF.md`

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `36 passed in 2.08s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/smoke_qdte.py`
  - Result: passed; smoke DP run completed and wrote `outputs/smoke_qdte_dp`.

### Current Status

- Implementation complete.
- Test verification complete.
- Smoke verification complete.

### Next Recommended Task

- Run a representative DP config and inspect `rho_spent`, debt diagnostics, no-active threshold logs, and prefix group monotonicity when `projection.prefix_monotonicity: true`.

## GPU Throughput Recovery Pass

### What Changed

- Kept the existing GPU candidate path active (`candidate_backend: jax_repair`) and added a cached GPU candidate context so static query/schema arrays are device-put once per run instead of every iteration.
- Padded active query ids to fixed `num_active_targets` capacity in the GPU path to avoid pmap recompiles when strict noise thresholding returns fewer active queries late in a run.
- Added `gpu_batches_per_iter` so a single QDTE iteration can run multiple GPU scoring microbatches without allocating one huge OOM-prone batch.
- Changed dense GPU scoring to use boolean enter/exit masks instead of materializing a float32 delta matrix, preserving the same edit-advantage formula while reducing temporary tensor pressure.
- Added query-block dense scoring (`gpu_score_query_block_size`) so the fused GPU path accumulates the same edit advantage over query blocks instead of forcing one huge candidate-by-query temporary matrix.
- Raised highpower throughput settings to `total_candidates_per_iter: 1572864`, `gpu_batches_per_iter: 1`, `gpu_score_query_block_size: 1024`, `gpu_return_top_k: 8192`, and `accepted_per_iter: 1024`.
- Set `allow_below_noise_fallback: true` only in `configs/adult_qdte_gpu_highpower.yaml` so the highpower profile can keep feeding GPUs; the ordinary adult config remains strict with fallback disabled.
- Added a small unit test for fixed-shape active query id padding.
- Updated README with the current highpower benchmark and the measured GPU power range.

### Changed Files

- `qdte/evolution/gpu_candidates.py`
- `qdte/evolution/scoring.py`
- `qdte/evolution/engine.py`
- `configs/adult_qdte_gpu_highpower.yaml`
- `tests/test_gpu_candidates.py`
- `README.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Benchmarks Run

- `configs/adult_qdte_gpu_highpower.yaml`, `max_iters=100`, `total_candidates_per_iter=262144`, `gpu_return_top_k=8192`, no final true-query eval:
  - Result: `26,214,400` candidates scored.
  - Runtime throughput: about `1.21M candidates/s`.
  - Sampled GPU power during sustained scoring: about `190-214W` per RTX 4090.
- `gpu_batches_per_iter=2`, `accepted_per_iter=1024`, `max_iters=50`:
  - Result: `26,214,400` candidates scored and `36,533` accepted edits.
  - Runtime throughput: about `1.21M candidates/s`.
  - Sampled GPU power during sustained scoring: about `190-212W` per RTX 4090.
- Tried `total_candidates_per_iter=524288`:
  - Result: failed with GPU OOM, about `21.12GiB` requested per card.
- Tried `total_candidates_per_iter=393216`:
  - Result: ran, but throughput was lower than the 262k setting and power stayed near `210W`.
- Tried `gpu_batches_per_iter=4`:
  - Result: ran, but did not improve candidates/s or sampled power over `gpu_batches_per_iter=2`.
- After boolean enter/exit scoring, tried `total_candidates_per_iter=524288`, `gpu_batches_per_iter=1`, `max_iters=50`:
  - Result: `26,214,400` candidates scored and `37,849` accepted edits.
  - Runtime scoring throughput: about `1.34M candidates/s`.
  - Sampled GPU power during sustained scoring: about `205-228W` per RTX 4090.
- Tried `total_candidates_per_iter=655360`, `gpu_batches_per_iter=1`, `max_iters=50`:
  - Result: `32,768,000` candidates scored and `36,609` accepted edits.
  - Runtime scoring throughput: about `1.38M candidates/s`.
- Tried `total_candidates_per_iter=786432`, `gpu_batches_per_iter=1`, `max_iters=50`:
  - Result: `39,321,600` candidates scored and `37,295` accepted edits.
  - Runtime scoring throughput: about `1.50M candidates/s`; overall scored throughput about `1.42M candidates/s`.
  - Sampled GPU power during sustained scoring: about `207-219W` per RTX 4090.
- Tried `total_candidates_per_iter=917504` and `1048576`:
  - Result: both failed with GPU OOM in JAX/XLA allocation, so they are not used as defaults.
- After query-block dense scoring, tried `total_candidates_per_iter=1048576`, `gpu_score_query_block_size=2048`, `max_iters=50`:
  - Result: `52,428,800` candidates scored and `37,243` accepted edits.
  - Runtime scoring throughput: about `1.82M candidates/s`.
  - Sampled GPU power during sustained scoring: about `220-242W` per RTX 4090.
- Tried `total_candidates_per_iter=1572864`, `gpu_score_query_block_size=1024`, `max_iters=50`:
  - Result: `78,643,200` candidates scored and `36,117` accepted edits.
  - Runtime scoring throughput: about `2.68M candidates/s`; overall scored throughput about `2.54M candidates/s`.
  - Sampled GPU power during sustained scoring: about `290-328W` on GPU 0 and `289-309W` on GPU 1.
- Tried `total_candidates_per_iter=2097152`, `gpu_score_query_block_size=1024`, `max_iters=50`:
  - Result: ran, but scoring throughput fell to about `2.26M candidates/s`, so it is not the default.

### Tests Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `40 passed in 2.55s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/smoke_qdte.py`
  - Result: passed; smoke DP run completed.

### Current Status

- GPU path confirmed present and active.
- Highpower config now feeds `1572864` scored candidates per iteration on the fused GPU path.
- Added real GPU scoring code changes, not just config changes.
- Power is recovered from the reported ~80W range to the requested ~300W/card range in local sustained sampling.

### Next Recommended Task

- Run a longer highpower DP experiment with final evaluation enabled and compare quality/runtime against the default quality-oriented config.

## Atom-Flow Transport

### What Changed

- Added `transport_mode: atom_flow` as a first-class transport mode.
- Implemented exact successive atom-flow transport over full encoded old-row/new-row atoms:
  - positive returned candidates are converted into atom edges;
  - edge units retain concrete candidate row ids;
  - each synthetic row has capacity one per transport batch;
  - each accepted flow unit updates the current weighted residual;
  - affected edge marginals are updated exactly through a query-to-edge inverted index;
  - the final batch is checked with the original QDTE batch advantage formula before application.
- Added atom-flow diagnostics to `metrics_timeseries.csv`:
  - `atom_flow_pool_candidates`
  - `atom_flow_edges`
  - `atom_flow_source_atoms`
  - `atom_flow_target_atoms`
  - `atom_flow_augments`
- Added last atom-flow diagnostics and atom-flow config fields to `runtime.json`.
- Allowed `qdte.transport_mode: atom_flow` in config validation.
- Switched `configs/adult_qdte.yaml` to quality-oriented `transport_mode: atom_flow`.
- Initially kept `configs/adult_qdte_gpu_highpower.yaml` on `microbatch_greedy` because exact atom-flow over large GPU returned pools moved the bottleneck to CPU transport. This is superseded by the later Batch Atom-Flow GPU Transport section.
- Updated README and architecture docs to describe atom-flow semantics and the highpower tradeoff.

### Changed Files

- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `configs/adult_qdte.yaml`
- `configs/adult_qdte_gpu_highpower.yaml`
- `tests/test_transport.py`
- `tests/test_config_validation.py`
- `tests/test_engine_smoke.py`
- `README.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests And Probes Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_transport.py tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - Result: `17 passed in 2.20s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `44 passed in 3.07s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/smoke_qdte.py`
  - Result: passed; standard smoke DP run completed.
- `configs/smoke.yaml` with `--qdte.transport_mode atom_flow --qdte.atom_flow_pool_multiplier 0 --debug.recompute_after_batch true --debug.assert_batch_loss_decrease true`
  - Result: passed; final measured loss `102.38`, accepted edits `772`, drift `0`.
- `configs/adult_qdte_gpu_highpower.yaml` with `--qdte.transport_mode atom_flow --qdte.atom_flow_max_pool 2048 --qdte.max_iters 3`
  - Result: ran; accepted `2925` edits, but `time_transport_seconds` was `32.55s`, so full/high-pool atom-flow is not the highpower default.
- `configs/adult_qdte_gpu_highpower.yaml` with `--qdte.transport_mode atom_flow --qdte.atom_flow_max_pool 1024 --qdte.max_iters 3`
  - Result: ran; accepted `1851` edits, `time_transport_seconds` was `12.53s`.
- `configs/adult_qdte_gpu_highpower.yaml` with `--qdte.transport_mode atom_flow --qdte.atom_flow_max_pool 512 --qdte.max_iters 3`
  - Result: ran; accepted `1117` edits, `time_transport_seconds` was `6.38s`.

### Current Status

- Atom-flow is implemented and tested as an exact successive marginal transport over atom edges, not a renamed prefix-greedy path.
- The DP boundary is preserved: atom-flow consumes only candidate deltas, noisy/projected residual, inverse variances, and edit costs.
- Quality-oriented Adult config now uses atom-flow.
- Exact atom-flow remains available with `qdte.atom_flow_update_mode: exact` for small-pool comparisons. The current highpower default is superseded by batch atom-flow below.

### Next Recommended Task

- Run a longer Adult atom-flow DP experiment with held-out evaluation enabled and compare `atom_flow_update_mode: batch` against `atom_flow_update_mode: exact` on measured loss, held-out true-query error, transport time, and accepted edits per scored candidate.

## Batch Atom-Flow GPU Transport

### What Changed

- Added `choose_atom_flow_batch_transport` as the default atom-flow update mode.
- Preserved the existing exact successive atom-flow path as `qdte.atom_flow_update_mode: exact`.
- Added `qdte.atom_flow_update_mode` config validation with allowed values `batch` and `exact`.
- Batch atom-flow now:
  - builds the same positive returned-candidate pool;
  - computes dense candidate deltas and quadratic terms on JAX, reusing the engine/scoring candidate advantages;
  - groups candidates by full encoded `old_row -> new_row` atom edges;
  - estimates same-edge diminishing-return capacity without updating residual after every flow unit;
  - enforces row id capacity once for the proposed batch;
  - uses JAX prefix objective evaluation to accept a positive batch prefix;
  - lets the engine update `answer_syn` and `residual` once after the accepted batch.
- Added diagnostics:
  - `atom_flow_batch_mode`
  - `atom_flow_exact_mode`
  - `atom_flow_selected_candidates`
  - `atom_flow_prefix_candidates`
  - `last_atom_flow_batch_mode`
  - `last_atom_flow_exact_mode`
- Switched both Adult configs to `transport_mode: atom_flow` with `atom_flow_update_mode: batch`.
- Updated README and architecture docs to distinguish batch/JAX atom-flow from exact atom-flow.

### Changed Files

- `qdte/evolution/transport.py`
- `qdte/evolution/engine.py`
- `qdte/config_validation.py`
- `configs/adult_qdte.yaml`
- `configs/adult_qdte_gpu_highpower.yaml`
- `tests/test_transport.py`
- `tests/test_config_validation.py`
- `tests/test_engine_smoke.py`
- `README.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests And Probes Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_transport.py tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - Result: `19 passed in 2.62s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `46 passed in 2.17s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/smoke_qdte.py`
  - Result: passed; standard smoke DP run completed. The current shell printed a JAX CUDA no-device plugin warning and then used CPU JAX devices for this smoke.
- `configs/adult_qdte_gpu_highpower.yaml` with `--qdte.max_iters 3 --qdte.log_every 1 --debug.recompute_after_batch true --debug.assert_batch_loss_decrease true`
  - Result: passed on two CUDA devices.
  - Scored `4718592` candidates.
  - Accepted `3072` edits.
  - `time_transport_seconds`: `1.3622361090019695`.
  - `time_scoring_seconds`: `9.67152326300129`.
  - Last returned pool: `16384`.
  - Last atom-flow edges: `9651`.
  - Final measured loss: `2.29923e+06`.
  - Drift: `0`.

### Current Status

- Batch atom-flow is now the default atom-flow path and highpower config path.
- Exact atom-flow remains implemented for audit/quality comparisons with `qdte.atom_flow_update_mode: exact`.
- Optimization invariants are preserved: the accepted prefix is still checked with the original measured-loss batch advantage formula, and the engine updates `residual = target - answer_syn` after applying the batch.
- DP boundary is preserved: batch atom-flow uses only noisy/projected residuals, inverse variances, candidate deltas, and edit costs.

### Next Recommended Task

- Run a longer Adult highpower batch atom-flow experiment with held-out evaluation enabled, then compare against exact atom-flow at a capped pool and microbatch greedy on measured loss, held-out true-query error, accepted edits, and transport/scoring time.

## Batch Atom-Flow Fine-Grained Optimization And 3-Round Probe

### What Changed

- Removed one redundant full-pool atom-edge counting pass from batch atom-flow; edge/source/target diagnostics are now computed from the same grouping pass used for selecting flow units.
- Stopped recomputing full candidate advantages inside batch atom-flow. The transport path now reuses the already-scored `advantages[pool_indices]` from the engine/scoring path and only computes `delta^2 @ inv_variance` on JAX for same-edge diminishing-return estimates.
- Kept the accepted prefix check unchanged: batch acceptance still uses the full QDTE batch objective before residual is updated once.
- Updated README and architecture docs with the current batch atom-flow behavior and latest 3-round probe timing.

### Changed Files

- `qdte/evolution/transport.py`
- `README.md`
- `architecture_zh.md`
- `docs/HANDOFF.md`

### Tests And Probes Run

- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest tests/test_transport.py tests/test_config_validation.py tests/test_engine_smoke.py -q`
  - Result: `19 passed in 1.65s`
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q`
  - Result: `46 passed in 2.10s`
- Three full Adult highpower 3-iteration probes with `--debug.recompute_after_batch true --debug.assert_batch_loss_decrease true`:
  - Round 1, seed 0: passed; `time_transport_seconds=1.0193166299977747`, final measured loss `2299220.2563900165`, accepted edits `3072`, drift `0`.
  - Round 2, seed 1: passed; `time_transport_seconds=1.0892132830012997`, final measured loss `2243135.423174282`, accepted edits `3072`, drift `0`.
  - Round 3, seed 2: passed; `time_transport_seconds=1.0204879980010446`, final measured loss `2279974.619533006`, accepted edits `3072`, drift `0`.
- `/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/smoke_qdte.py`
  - Result: passed; standard smoke DP run completed. This shell printed the known JAX CUDA no-device warning and then used CPU JAX devices for the smoke path.

### Issues Found

- No objective-invariant, residual-drift, GPU-table-sync, or test failures were found in the 3-round probe.
- Short 3-iteration probes are not reliable evidence of true-query generalization: rounds 2 and 3 reduced measured loss but had final true-query MAE slightly above initial true-query MAE. This is an evaluation-risk signal, not an optimization-loop privacy violation.

### Current Status

- Fine-grained batch atom-flow optimization is implemented and verified.
- Current highpower transport time for the 16k returned-candidate pool is about `1.02-1.09s` over 3 iterations.
- QDTE optimization behavior remains bounded by noisy/projected residuals, inverse variances, candidate deltas, and edit costs.

### Next Recommended Task

- Run a longer held-out evaluation experiment for batch atom-flow, then compare measured-loss improvement against held-out true-query MAE/RMSE to quantify whether the aggressive highpower path overfits DP-noisy measured queries.

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
