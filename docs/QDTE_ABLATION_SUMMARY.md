# QDTE Ablation Summary

This is a living registry of QDTE ablations discussed and prototyped during development.
Use it to plan the final full experiment matrix. Do not treat historical numbers as
paper-ready until the corresponding row is refreshed under one frozen commit/config.

For the run list, seed policy, A/B/C/D route plan, and final reporting checklist,
see `docs/QDTE_FINAL_EXPERIMENT_PLAN.md`.

## Core Question

QDTE should be evaluated along three axes:

- Proposal source: random individual edits, residual-directed edits, masked edits, enumerated local edits, or constructed partners.
- Acceptance object: single edit, pair, random group, or directed group.
- Scoring rule: unchanged QDTE measured-loss advantage using noisy/projected targets and variances.

The most important invariant is that active optimization in `privacy.mode=dp` uses only:

- projected/noisy measurements;
- residuals against those measurements;
- inverse variances;
- candidate deltas;
- edit costs.

Exact true answers are evaluation-only.

## Current-Code Reference Runs

All rows below were run on `configs/smoke.yaml`, seed `0`, 2000 iterations, with offline true-query evaluation.
Most rows used 512000 scored candidates; rows with side-budget partner search state their actual scored-candidate count in the notes.
These rows did **not** enable `projection.consistency`; their measured losses are comparable to each other, but not to the older
`local_table_feasible_jax` A/B runs.

| Variant | Proposal | Acceptance object | Final measured loss | True RMSE | True MAE | Accepted edits | Wall seconds | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `random_best_edit` | random edit | best single edit | 66.624540 | 0.004038 | 0.002523 | 1150 | 4.61 | `outputs/exp_random_best_current_2000_random_best_edit` |
| `random_group_advantage`, size `2..8`, 128 groups/iter | random edit | random group | 82.419445 | 0.005269 | 0.002918 | 2842 | 8.66 | `outputs/exp_random_group_2000_random_group_advantage` |
| `random_group_advantage`, size `1..8`, 512 groups/iter | random edit | random group | 68.857956 | 0.004230 | 0.002634 | 2180 | 18.21 | `outputs/exp_random_group_min1_2000_random_group_advantage` |
| `random_group_advantage`, size `1`, 512 groups/iter | random edit | sampled single edit | 67.231262 | 0.004130 | 0.002494 | 1160 | 16.31 | `outputs/exp_random_group_size1_2000_random_group_advantage` |
| `directed_group_advantage`, max size `2` | random edit | residual/delta-directed group | 59.628688 | 0.003222 | 0.002202 | 1655 | 11.12 | `outputs/exp_directed_group_g2_2000_directed_group_advantage` |
| `directed_group_advantage`, max size `4` | random edit | residual/delta-directed group | 59.624552 | 0.003205 | 0.002148 | 1843 | 13.72 | `outputs/exp_directed_group_g4_2000_directed_group_advantage` |
| `directed_group_advantage`, max size `8` | random edit | residual/delta-directed group | 59.627354 | 0.003232 | 0.002202 | 2035 | 14.47 | `outputs/exp_directed_group_g8_2000_directed_group_advantage` |
| `protected_same_row` C | protected same-row repair | constructive pair transport | 73.254116 | 0.004695 | 0.002712 | 1195 | 70.03 | `outputs/exp_cd_pilot_2000_protected_same_row` |
| `bounded_best_partner` D | bounded best-partner search, side budget 128 | attached pair plus constructive pair transport | 59.726185 | 0.003227 | 0.002185 | 1239 | 802.98 | `outputs/exp_cd_pilot_2000_densecpu_bounded_best_partner`; scored 522460 candidates |

Current interpretation:

- Pure random group scoring does not beat current random best-single edit on this setup.
- Allowing group size `1` mostly collapses back to best-single behavior.
- Random groups have very low positive aggregate-advantage hit rate late in optimization.
- Residual/delta-directed group construction beats both current random best-single and random group variants on this setup.
- Group size matters, but max size `2`, `4`, and `8` are close here; max size `4` is the best of this sweep by measured loss and true RMSE.
- C `protected_same_row` is still too constrained: it improves early but stalls well above the stronger directed/group routes.
- D `bounded_best_partner` validates the best-partner direction, but in this pilot it is only close to directed group and is far slower because candidate generation dominates runtime.

## Local-Table Feasible Projection Check

The older A/B/random-best numbers used `projection.consistency.enabled=true` and
`projection.consistency.method=local_table_feasible_jax`. That projection changes
the optimized target, so measured loss is on a different scale from the current
no-consistency table above.

Current code reproduces the old `random_best_edit` result when the same
projection settings are enabled. C and D were also rerun under this same
projected target, because their no-consistency pilot losses are not comparable
to the older A/B runs.

| Variant | Projection | Final measured loss | True RMSE | True MAE | Accepted edits | Scored candidates | Generation seconds | Output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `random_best_edit` | none | 66.624540 | 0.004038 | 0.002523 | 1150 | 512000 | 4.61 | `outputs/exp_random_best_current_2000_random_best_edit` |
| `random_best_edit` | `local_table_feasible_jax` | 7.243474 | 0.003833 | 0.002502 | 1175 | 512000 | n/a | `outputs/exp_random_best_consistency_check_2000_random_best_edit` |
| `constructive_pair` A fixed-8 | `local_table_feasible_jax` | 2.213864 | 0.003275 | 0.002267 | 1535 | 512000 | 15.75 | `outputs/exp_a_current_consistency_2000_constructive_pair` |
| `constructive_pair_accept_anneal32` A | `local_table_feasible_jax` | 1.932743 | 0.003295 | 0.002243 | 1750 | 512000 | 16.16 | `outputs/exp_a_current_consistency_2000_constructive_pair_accept_anneal32` |
| `protected_same_row` C | `local_table_feasible_jax` | 11.509982 | 0.004220 | 0.002621 | 1140 | 512000 | 73.00 | `outputs/exp_cd_consistency_2000_protected_same_row` |
| `bounded_best_partner` D | `local_table_feasible_jax` | 1.997482 | 0.003235 | 0.002218 | 1169 | 520994 | 116.45 | `outputs/exp_cd_consistency_2000_bounded_best_partner` |
| `bounded_best_partner_cached` D | `local_table_feasible_jax` | 2.021055 | 0.003294 | 0.002284 | 1166 | 520795 | 82.82 | `outputs/exp_cd_consistency_2000_cached_bounded_best_partner_cached` |
| `bounded_best_partner_gpu` fast D | `local_table_feasible_jax` | 2.153643 | 0.003305 | 0.002288 | 1120 | 520131 | 73.98 | `outputs/exp_cd_consistency_2000_jaxbatch_bounded_best_partner` |

Interpretation:

- The apparent regression from `7.24` to `66.62` was a projection-target mismatch, not an algorithm regression.
- `local_table_feasible_jax` produces a feasible/nonnegative projected target that can be fit much more tightly by synthetic microdata.
- No-consistency noisy measurements can be mutually inconsistent, so the same optimizer has a much higher measured-loss floor.
- C remains weaker than A/B under the fair projected target, but the no-consistency C loss was not the right evidence for that conclusion.
- D changes materially under the fair projected target: it beats fixed-8 A by measured loss on this seed (`1.997482` vs `2.213864`) and is close to annealed A (`1.932743`), but costs about `7.4x` A's generation time and scores an extra side-budget of partner candidates.
- `bounded_best_partner_cached` caches current-source row query values for D partner delta computation. It reduces 2000-step generation time from `116.45s` to `82.82s` and candidate-generation time from `82.37s` to `48.39s`. It reproduced dense D exactly in the 100-step probe, but the 2000-step path diverged slightly through tiny numerical/tie effects, landing at loss `2.021055` instead of `1.997482`.
- `bounded_best_partner_gpu` is a fast D variant, not a bitwise-equivalent replacement for D. It batches partner delta evaluation on JAX/GPU and reduces 2000-step generation time from `116.45s` to `73.98s` and candidate-generation time from `82.37s` to `27.27s`, but its seed-chunk partner search changes the stochastic path and this run lands at measured loss `2.153643`.
- Strict-equivalence probes did not improve speed on this smoke setup:
  - `bounded_best_partner_source_cached` reproduced the 100-step dense D result but had candidate-generation time `21.79s` vs dense `21.60s`.
  - `bounded_best_partner_unique` reproduced the 100-step dense D result but had candidate-generation time `22.55s` vs dense `21.60s`.
- Future full comparisons must freeze the projection setting and should not mix the two measured-loss scales.
- Current code reproduces A's earlier smoke results under the same feasible projection.
  A is much faster than D on this seed: A generation time is about `16s`, while
  dense D takes `116.45s`, cached D takes `82.82s`, and JAX-batch D takes
  `73.98s`. A anneal32 is also the best measured-loss row in this projected
  smoke table (`1.932743`), while dense D has slightly better offline true
  RMSE/MAE (`0.003235`/`0.002218` vs A anneal32's
  `0.003295`/`0.002243`).

### A Under The D-Style Narrative

D is the clearest record-level story: start from an edit that helps the residual
objective, identify the queries it harms, then search for a partner edit whose
delta repairs those harms while preserving aggregate advantage.

A can be described as the efficient pool-based version of the same principle:
it first builds a residual-directed candidate pool, then compiles candidates
into pair/prefix bundles by exact aggregate edit advantage. A does not synthesize
a custom partner for each seed edit the way D does; instead, it opportunistically
finds compensating partners already present in the directed pool. This makes A
less explicit as an explanation of "what should this exact record become," but
much more practical computationally.

The recommended paper framing is therefore:

- The core idea is individual-level directed edit-bundle construction, scored by
  exact aggregate QDTE edit advantage.
- A is the fast implementation: generate a directed candidate pool once per
  iteration and select aggregate-advantage bundles from it.
- D is the explicit/interpretable instantiation: for a seed edit, construct or
  search a partner edit targeted at harmed queries.
- If final full experiments keep matching this smoke result, A should be the
  main practical algorithm and D should be used as the interpretability and
  mechanism ablation.

## Historical Exploratory Runs

These results are useful for hypothesis formation, but should be refreshed before final reporting.
They may have been run before later implementation changes.

| Variant | Main idea | Historical final measured loss | True RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| `constructive_pair` A fixed-8 | residual-directed candidates plus aggregate pair/prefix advantage | 2.213864 | 0.003275 | Reproduced under current code with `local_table_feasible_jax`; fast pool-based counterpart to D's explicit partner story. |
| `constructive_pair_accept_anneal32` | A with accepted edit schedule 32 -> 2 | 1.932743 | 0.003295 | Reproduced under current code with `local_table_feasible_jax`; best current projected smoke measured loss. |
| `constructive_partner_b2` | keep A pool, attach synthesized partner side pool | 2.479907 | 0.003322 | Slower than A; current-code refresh required. |
| `protected_same_row` C | same-row protected repair for harmed queries | 11.509982 | 0.004220 | Reproduced under current code with `local_table_feasible_jax`; generated candidates but became too constrained late. |
| `bounded_best_partner` D | bounded best-partner search | 1.997482 | 0.003235 | Current-code `local_table_feasible_jax` rerun; close to/better than A fixed-8 on seed `0`, but slower and slightly worse than annealed A. |
| `bounded_best_partner_cached` D | cached current-source phi for partner scoring | 2.021055 | 0.003294 | Faster close variant; not a strict bitwise replacement for dense D over 2000 steps. |
| `bounded_best_partner_gpu` fast D | seed-chunked JAX/GPU partner scoring | 2.153643 | 0.003305 | Faster D implementation variant; useful for performance sweeps, but keep separate from dense D quality claims. |
| `random_best_edit` with `local_table_feasible_jax` | random one-row mutation plus best single edit | 7.243474 | 0.003833 | Reproduced under current code; compare only against other runs with the same projection setting. |
| old `random_mutation` | random mutation, multi-edit acceptance | 65.748090 | 0.003896 | Similar scale to current random best-single. |

## Ablation Registry

### Random Proposal Baselines

- `random_mutation`
  - Purpose: random individual edits with default multi-edit acceptance.
  - Use as a weak random mutation baseline.
- `random_best_edit`
  - Purpose: random individual edits, accept best single edit.
  - Use as the stronger random-search baseline.
- `pgsd_style_mutate50`
  - Purpose: closer mutate-only Private-GSD-style budget with 50 random candidates/iter.
  - Not an exact upstream Private-GSD run.

### Directed Single-Edit Proposal Variants

- `single_query`
  - Residual-selected query chooses enter/exit direction for one record edit.
- `masked_single_query`
  - Masked target query construction.
  - Earlier experiments suggested it can narrow the useful search too much.
- `relaxed_masked_single_query`
  - Relaxed mask variant intended to expand masked search.
- `masked_exit_query`, `masked_exit_only`, `directed_exit_only`
  - Exit-side directed variants.
- `residual_weighted_mutation`, `residual_value_mutation`
  - Residual-weighted value/attribute sampling.
- `enumerated_local`
  - Local enumeration around records.
  - Historically sometimes strong, but can be slower and less conceptually clean.
- `proposal_mixture`, `qdte_mixture`
  - Mixtures of directed/random/enumerated proposal families.

### Aggregate Advantage Variants

- `constructive_pair`
  - Scores two-edit units and accepted prefixes with exact aggregate advantage.
  - This is the main historical evidence that aggregate edit advantage helps.
- `constructive_pair_accept_anneal`, `constructive_pair_accept_anneal32`
  - Same pair construction, but annealed accepted edit count.
- `constructive_pair_mixture`
  - Pair transport over mixture proposal pool.
  - Historical result was worse than pure A.
- `random_group_advantage`
  - Randomly sample non-conflicting edit groups and score exact aggregate advantage.
  - Current result: not enough by itself; group construction needs direction.
- `directed_group_advantage`
  - Residual/delta greedy group construction over random individual edit proposals.
  - Current sweep over max group size `2`, `4`, and `8` improves over random baselines.
  - The selected best group is often smaller than the allowed max size; keep max size as a sweep parameter.

### Constructive Partner / Repair Variants

- `constructive_partner`
  - Prototype: synthesize partner candidates for harmed query compensation.
- `constructive_partner_b2`
  - Preserves A's seed pool and attaches side-budget partners.
  - Historically did not beat A, but still validates explicit pair scoring.
- `protected_same_row`
  - Same-row target/protection repair.
  - Standalone version is not competitive; useful as a diagnostic and possible side-budget fallback.
- `bounded_best_partner`
  - Bounded D-style partner search.
  - For each seed, samples partner sources for harmed queries, generates multiple repairs, scores full `delta_seed + delta_partner` with the QDTE objective, and attaches only the best positive partners.
  - Current pilot is much slower than A/group methods and does not beat directed group on seed `0`; keep it as an oracle/diagnostic unless future tuning changes the tradeoff.

## Recommended Final Matrix

Run under one frozen commit/config, multiple seeds:

- `random_best_edit`
- `random_group_advantage` with sizes:
  - fixed `1`
  - fixed `2`
  - `1..8`
  - `2..8`
- `directed_group_advantage` with `directed_group_max_size`:
  - `1`
  - `2`
  - `4`
  - `8`
  - optionally `16` if runtime is acceptable
- `constructive_pair`
- `constructive_pair_accept_anneal32`
- `constructive_partner_b2`
- `protected_same_row`
- exact upstream Private-GSD or a clearly labeled compatible wrapper if paper-level comparison is needed.

For every run report:

- final measured loss;
- RMS standardized residual;
- true-query MAE/RMSE for offline evaluation;
- held-out query MAE/RMSE when enabled;
- accepted edits;
- candidates scored;
- wall time and transport time;
- group/pair diagnostics when applicable.

## Working Interpretation

The strongest story is not "group advantage alone works." The random group probe shows that group-level scoring without directed construction has low hit rate.

The more defensible story is:

- edit advantage is the exact local measured-loss drop;
- single-edit scoring is too myopic;
- aggregate edit advantage is useful when the group is built with residual/delta structure;
- QDTE's distinguishing direction is individual-level, residual-guided edit or edit-bundle construction inside one synthetic dataset, not only population-level random dataset search.
