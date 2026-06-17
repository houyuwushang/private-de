# QDTE Final Experiment Plan

This document lists the experiments that should be run before paper-level
reporting. It is intentionally a plan, not a result table. Smoke runs and
historical exploratory results live in `docs/QDTE_ABLATION_SUMMARY.md`.

## Goal

The final experiments should answer four questions:

- Does QDTE's residual-directed individual edit field improve over random
  mutation and Private-GSD-style dataset-level random search?
- Which part matters: directed proposal generation, edit advantage, aggregate
  advantage, or constructive residual/delta coordination?
- Is the current A route, pool-level constructive pairing, already the best
  accuracy/complexity tradeoff?
- Do C or D add enough over A to justify their extra implementation and runtime
  complexity?

All DP-mode optimization must use only noisy/projected measurements, variances,
residuals, candidate deltas, and edit costs. Exact true answers are evaluation
only and must not drive active selection, candidate generation, scoring,
transport, stopping, or hyperparameter selection.

## Freeze Before Full Runs

Before running the final matrix:

- freeze one git commit and record it in every run artifact;
- freeze one config family for each dataset/scale;
- cache or otherwise reuse identical DP measurement/projection targets when
  comparing algorithm variants;
- use fixed public metadata, workload, privacy parameters, and synthetic row
  count across variants;
- choose hyperparameters from pilot measured-loss runs only, not true-query
  metrics;
- record wall time, candidate generation time, scoring time, transport time,
  and GPU/CPU backend.

Recommended seed policy:

- pilot: seeds `0,1,2`;
- final ablation: seeds `0..4` at minimum;
- final main table: seeds `0..9` if runtime is acceptable.

Recommended budget policy:

- report by candidate evaluations as the primary compute budget;
- also report iterations because Private-GSD-style methods often use many
  small population-level steps;
- include at least one long-run budget where A and the strongest baselines are
  no longer improving quickly.

## Main Paper Comparison

Run these under the same DP target, workload, row count, and candidate-evaluation
budget.

| Label | Current variant or source | Purpose |
| --- | --- | --- |
| Upstream Private-GSD | external baseline wrapper | Paper-level population random mutation baseline. |
| Private-GSD-style mutate | `pgsd_style_mutate50` or exact wrapper | Internal sanity baseline if upstream integration is costly. |
| Random mutation | `random_mutation` | Random proposals plus normal QDTE acceptance. |
| Random best edit | `random_best_edit` | Strong random single-edit baseline. |
| Directed single edit | `single_query` | Basic individual-level residual-directed edit field. |
| QDTE mixture | `qdte_mixture` | Broad-support directed proposal mixture. |
| A: constructive pair | `constructive_pair` | Current leading route: pair existing directed edits by residual/delta compensation. |
| A with annealed accept count | `constructive_pair_accept_anneal32` | Tests whether early larger batches and late smaller batches improve A. |
| Directed group advantage | `directed_group_advantage` | Residual/delta group construction over random proposals. |
| B2: attached partner | `constructive_partner_b2` | Tests whether synthesized partner candidates beat pool-only A. |
| C: protected same-row repair | `protected_same_row` | Tests explicit same-row protection of harmed predicates. |
| D: bounded best partner | `bounded_best_partner` | Bounded diagnostic for whether best partner search can improve A/B2. |

Main metrics:

- final measured loss;
- RMS standardized residual;
- true-query MAE/RMSE for offline evaluation;
- held-out query MAE/RMSE when enabled;
- accepted edits and accepted units;
- candidate evaluations;
- wall time and phase timings;
- peak memory / backend notes.

## Core Ablation Matrix

### Proposal Direction

Purpose: show whether residual-directed proposal generation is useful beyond
random mutation.

Run:

- `random_mutation`
- `random_best_edit`
- `single_query`
- `soft_single_query`
- `residual_weighted_mutation`
- `residual_value_mutation`
- `enumerated_local`
- `proposal_mixture`
- `qdte_mixture`

Report:

- final loss and convergence curves;
- candidate shortfall and fallback-random counts;
- active-target residual signs selected by each method;
- accepted edit types.

### Mask And Paired Proposal Variants

Purpose: show why earlier mask/paired routes are not the main method unless
they recover under full-scale runs.

Run:

- `masked_single_query`
- `relaxed_masked_single_query`
- `paired_query`
- `paired_query_full`
- `masked_paired_query`
- `masked_paired_query_full`
- `masked_exit_query`
- `masked_exit_query_full`
- `directed_exit_only`
- `masked_exit_only`
- `random_source_directed_exit`

Report:

- final loss;
- candidate diversity;
- source-filter failure rate;
- whether masks increase or reduce useful proposal support;
- blind-loss controls when feasible.

### Edit-Advantage Controls

Purpose: separate candidate generation from the QDTE acceptance objective.

Run the strongest relevant variants with normal acceptance and `blind_` prefix:

- `blind_random_mutation`
- `blind_single_query`
- `blind_masked_single_query`
- `blind_relaxed_masked_single_query`
- `blind_enumerated_local`
- `blind_constructive_pair`
- `blind_constructive_partner_b2`
- `blind_protected_same_row`

Interpretation:

- If blind versions collapse, edit advantage is doing necessary selection.
- If a blind directed variant is strong, the proposal itself is carrying more
  of the optimization than expected.
- True-query metrics remain offline only.

### Aggregate Advantage And Group Size

Purpose: test whether multi-edit scoring helps and whether it needs directed
construction.

Run:

- `random_group_advantage`, fixed group size `1`;
- `random_group_advantage`, fixed group size `2`;
- `random_group_advantage`, size `1..4`;
- `random_group_advantage`, size `1..8`;
- `random_group_advantage`, size `2..8`;
- `directed_group_advantage`, max size `1`;
- `directed_group_advantage`, max size `2`;
- `directed_group_advantage`, max size `4`;
- `directed_group_advantage`, max size `8`;
- `directed_group_advantage`, max size `16` if runtime is acceptable.

Sweep:

- `directed_group_seed_count`: `16,32,64,128`;
- `directed_group_pool_multiplier`: `0,4,8` or equivalent max-pool caps;
- `directed_group_allow_negative_steps`: `false,true` as an exploratory
  rescue of individually bad but jointly useful edits.

Report:

- accepted group-size distribution;
- positive group hit rate;
- groups with negative members;
- late-stage stall iteration;
- transport time.

## A/B/C/D Route Plan

### A: Pool-Level Constructive Pair

Status: implemented as `transport_mode=constructive_pair` and currently the
leading historical route.

Full experiments:

- `constructive_pair`
- `constructive_pair_accept_anneal32`
- `constructive_pair_mixture`
- A with `constructive_pair_partner_limit`: `4,8,16,32`
- A with `constructive_pair_harm_query_limit`: `4,8,16,32`
- A with `constructive_pair_max_units` capped versus uncapped

Diagnostics:

- positive pair units per iteration;
- explicit attached-pair units if B2 is active;
- selected single units versus pair units;
- final plateau loss and last positive-pair iteration.

Decision:

- If A remains best or tied with simpler runtime, use A as the paper method.
- If A only wins with an impractically large pair-search budget, compare it by
  equal candidate-evaluation and equal wall-time budgets.

### B2: Attached Synthesized Partner

Status: implemented, but earlier smoke suggested it did not beat A.

Full experiments:

- `constructive_partner_b2`
- `constructive_partner_b2_accept_anneal32`
- side budget: `32,64,128,256`
- harmed queries per seed: `2,4,8`
- partners per seed: `1,2,4`
- source over-sample factor: `4,8,16`

Diagnostics:

- attached partner candidates;
- attached pair units;
- source attempts and failures;
- explicit positive pair rate;
- whether partners are accepted directly or only help ranking.

Decision:

- Keep B2 only if it improves A under fair compute or clearly reduces A's
  late-stage plateau.
- Otherwise describe B2 as a negative result showing that pool-level pair
  construction was already capturing most available compensation.

### C: Protected Same-Row Repair

Status: implemented as `protected_same_row`, but historical result was weaker
than A.

Why it matters:

- A and B use two-row compensation: one edit may harm a query and another edit
  cancels it.
- C tries to make one edit less harmful by rebuilding the destination while
  preserving predicates from harmed residual-weighted queries.
- This is the most direct follow-up to the earlier mask idea: the mask is no
  longer arbitrary; it is selected from the seed edit's harmful support.

Full experiments:

- current standalone `protected_same_row`;
- C as fallback after A/B2 source failure, if implemented;
- C with `protected_repair_harm_queries`: `2,4,8`;
- C with `protected_repair_restarts_per_seed`: `2,4,8,16`;
- C with `protected_repair_max_protection_passes`: `1,2,4`;
- C with and without `protected_repair_require_target_direction`;
- C plus constructive pair transport versus single-edit transport.

Diagnostics:

- protected repair attempts;
- target failures;
- protection successes;
- candidate count and accepted count;
- measured harmful-support reduction before/after repair;
- whether C improves late-stage positive-candidate hit rate.

Decision:

- C is worth keeping if it either beats A or works as a cheap fallback that
  improves A/B2 without narrowing proposal support too much.
- If standalone C remains weak and fallback C is marginal, keep it as a
  diagnostic/negative ablation rather than the main method.

### D: Exact Or Bounded Best Partner

Status: bounded diagnostic version implemented as `bounded_best_partner`.
It is not recommended as the first main route unless future runs show a clear
quality gain under fair compute.

Why it matters:

- D asks whether A is losing because the best compensating partner is absent
  from the finite candidate pool.
- If D cannot beat A even on small/bounded cases, A is likely close to the best
  practical route.
- If D beats A strongly, it tells us the missing piece is partner generation,
  not the aggregate advantage objective.

Possible implementations:

- Implemented bounded partner diagnostic:
  - generate the same seed edit pool as A;
  - for each seed, inspect harmed residual-weighted queries;
  - sample partner sources from the required opposite harmed-query side;
  - generate multiple repair restarts;
  - score full `delta_seed + delta_partner` with the QDTE objective;
  - attach only the best positive partner candidates to the constructive-pair
    transport.
- Small-scope oracle:
  - restrict to a tiny workload/domain or top harmed-query subset;
  - enumerate feasible partner sources and a bounded destination value set;
  - score exact aggregate advantage.
- Beam-search partner:
  - start from source rows satisfying the required opposite harmed-query sign;
  - repair one attribute at a time;
  - keep top beams by aggregate residual/delta score.
- Constraint-solver partner:
  - encode top harmed query predicates and domain constraints;
  - optimize the residual/delta objective over a bounded candidate set;
  - use only as a diagnostic if runtime is high.

Full experiments if D is implemented:

- current `bounded_best_partner` on smoke/small workload;
- D oracle on smoke/small workload only, if a stronger oracle is still needed;
- D bounded beam with beam sizes `4,8,16`;
- D top harmed-query counts `2,4,8`;
- D as pair transport against A under equal candidate-evaluation budget;
- D as upper-bound diagnostic where wall time is reported separately.

Diagnostics:

- partner source availability;
- destination feasibility;
- best partner advantage over A's best pool partner;
- solver/beam time;
- success rate by iteration phase.

Decision:

- If D only improves A on very small scopes or with much higher wall time, do
  not use it as the main paper method.
- If D materially beats A under fair budget, implement the cheapest
  approximation of D and compare it as the main constructive route.
- Current seed-0 pilot result:
  - `protected_same_row` C: measured loss `73.254116`, true RMSE `0.004695`,
    runtime about `70s`;
  - `bounded_best_partner` D: measured loss `59.726185`, true RMSE `0.003227`,
    runtime about `803s`, with `522460` scored candidates due side-budget
    partner search;
  - D is much better than C but does not beat the current `directed_group`
    pilot and is far slower, so it should remain diagnostic unless A refreshes
    weaker than expected.

## Projection And Measurement Ablations

Purpose: separate algorithm quality from target-projection choices.

Run the strongest algorithm subset under:

- no consistency projection, if supported;
- equality-only unbiased consistency projection;
- local-table feasible/nonnegative projection as a biased lower-MSE variant;
- JAX and CPU projection backends only when they are mathematically equivalent;
- continuous Gaussian DP noise;
- discrete Gaussian DP noise if implemented.

Strong subset:

- `random_best_edit`
- `single_query`
- `constructive_pair`
- `constructive_pair_accept_anneal32`
- `directed_group_advantage`
- `constructive_partner_b2`
- `protected_same_row`

Report:

- measured loss against the actual projected target used for optimization;
- true-query error for offline evaluation;
- negative projected answer counts for equality-only projection;
- feasibility constraint violation;
- projection runtime.

Important interpretation:

- Equality-only projection is the unbiased baseline but can produce negative
  answers.
- Nonnegative/local-table projection is biased but may lower overall MSE and
  may make the optimization target more feasible for synthetic microdata.
- Do not use true-query metrics to choose the projection hyperparameters in DP
  experiments.

## Select-Measure-Generate Framework Roadmap

The generation-algorithm exploration is far enough to freeze A as the first
practical generator candidate and move to a cleaner select-measure-generate
framework. The framework should separate the DP-sensitive measurement boundary
from post-processing generation:

- Select:
  - choose the measured workload or workload groups;
  - start with the existing static workload selection;
  - later add adaptive query/group selection only if the privacy accounting is
    explicit and the selected measurements consume budget correctly.
- Measure/project:
  - run DP measurement on the selected workload;
  - apply the chosen consistency projection;
  - persist the noisy/projected target, variances, workload, schema, and privacy
    accounting as a reusable measured target artifact.
- Generate:
  - initialize one or more synthetic datasets from the measured target;
  - run a selected generator against only the noisy/projected target and
    variances;
  - use A `constructive_pair_accept_anneal32` as the first main generator,
    with fixed-8 A and random baselines as controls.

Implementation target:

- add a reusable measured-target loading/saving boundary so multiple generator
  variants can consume exactly the same DP measurements;
- avoid remeasuring real data inside generator comparisons;
- keep exact true-query answers available only in offline evaluation outputs;
- report the selected workload, projection method, generator, and all runtime
  budgets separately.

## Outer Dataset-Level Evolution

The current QDTE generator is single-dataset evolution. The outer
dataset-level loop has not yet been implemented as a first-class algorithmic
layer. It should be added only as post-processing over one fixed measured
target, not as a new measurement mechanism.

Recommended design:

- Maintain a population of synthetic dataset states, each with its own
  `X_syn`, `answer_syn`, `residual`, and RNG stream.
- All individuals share the same measured target, variances, schema, and query
  catalogue.
- For each outer generation:
  - run the inner QDTE generator for a small fixed budget on each individual;
  - score individuals by measured loss only;
  - keep an elite set;
  - refill the population using restarts, cloned elites with fresh local
    mutation, or row/block crossover between strong individuals;
  - never use offline true-query metrics for parent selection, restart policy,
    stopping, or hyperparameter choice in DP runs.
- Compare by equal total candidate evaluations and equal wall time, because a
  population loop can otherwise look better simply by spending more generation
  budget.

Minimal variants:

- Inner-only A: current `constructive_pair_accept_anneal32`.
- Random-restart ensemble: run multiple A instances and keep the best measured
  loss at the end.
- Population A: run short inner A phases, select by measured loss, and clone or
  restart stalled individuals.
- Population random mutation: Private-GSD-style population control using the
  same measured target, as an internal baseline.

Research value:

- If population A improves over inner-only A under equal candidate-evaluation
  budget, the final algorithm can be described as a two-level method:
  individual-level directed edit-bundle evolution inside each dataset plus
  dataset-level evolutionary selection across synthetic states.
- If population A only helps by spending more compute, keep it as an engineering
  ensemble/robustness option rather than the central novelty.
- If population random mutation remains weaker than population A, it reinforces
  the key distinction from Private-GSD-style random dataset mutation: QDTE's
  inner generator is directed at the individual edit-bundle level.

## Scale And Robustness Experiments

Run the final small subset on larger or varied settings:

- row counts: small smoke, medium, and the largest affordable dataset setting;
- workload scope/order: low-order, mixed-order, and high-order query workloads;
- privacy budgets: at least three privacy levels, e.g. low/medium/high noise;
- candidate budgets: short, medium, long;
- seeds: at least five for ablations, ten for main table if feasible.

Final small subset:

- upstream Private-GSD or exact compatible wrapper;
- `random_best_edit`;
- `single_query`;
- A: `constructive_pair`;
- best A schedule if selected;
- best directed-group variant if it remains competitive;
- best C/D variant only if it beats or complements A.

## Reporting Checklist

Every final run should save:

- resolved config;
- git commit;
- command line;
- metrics timeseries;
- final metrics;
- runtime breakdown;
- candidate diagnostics;
- projection diagnostics;
- offline true-query and held-out metrics when enabled;
- note that true answers were evaluation-only in DP mode.

Every final table should state:

- dataset and public metadata;
- workload size and query family;
- privacy mode and privacy parameters;
- projection method;
- candidate-evaluation budget;
- iteration budget;
- seed count;
- whether results are equal compute or equal wall time.

## Current Working Decision Rule

Do not overbuild C/D before A is fairly tested. The likely paper choice is:

- use A if it remains best or nearly best under full runs;
- add A's annealed accept schedule if it improves consistently;
- keep B2/C/D as ablations or diagnostics unless they beat A under fair compute;
- describe the contribution as residual-directed individual edit construction
  plus aggregate residual/delta coordination;
- call the full method a two-level evolutionary algorithm only after the outer
  dataset-population loop is implemented and evaluated.
