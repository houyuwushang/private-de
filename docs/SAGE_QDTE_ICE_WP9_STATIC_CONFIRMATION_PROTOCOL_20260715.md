# SAGE-QDTE-ICE WP9 Static Confirmation Protocol

Date: 2026-07-15

Protocol ID:

```text
SAGE-QDTE-ICE-WP9-STATIC-CONFIRMATION-20260715-v1
```

## 1. Question

The mechanism search through WP8a leaves one stable positive method:

```text
full public Static-ICE measurement
+ exact orthogonal interaction precision
+ QDTE Standard row evolution
```

WP9 asks one confirmatory question:

> Does this frozen Static-ICE-Exact profile beat Official AIM reliably across
> all four datasets at epsilon 0.1 and 0.3, rather than only on inspected
> seed0 development cells?

This is a confirmation experiment, not another method-development sweep.

## 2. Why This Branch

WP8a established that private partition-L1 refinement has real allocation
signal but does not beat full Static-ICE average utility. Its frozen decision
is `stop_wp8a`.

The proposed full-coverage Official Private-PGM diagnostic is not tractable on
Adult. Measuring every pair induces a complete 15-attribute graph, whose
single maximal clique has:

```text
424,688,379,494,400 cells
```

This exceeds the frozen 20,000,000-cell resource cap by more than seven orders
of magnitude. Raising the cap or silently dropping interactions would not be
the declared full-transcript diagnostic. A released sparse-tree prior remains
a separate future candidate and is not part of WP9.

## 3. Frozen Method

Method name:

```text
SAGE-QDTE-Static-ICE-Exact-v1
```

Measurement:

```text
adjacency: unbounded / add-remove
delta_DP: 1e-9
public n: required
public schema/cardinalities: required
one-way blocks: every public attribute Helmert contrast block
pair blocks: every public pair satisfying product(cardinalities) <= 20,000
measurement: independent Gaussian orthogonal coefficient vectors
allocation: public_optimal
projection adapter: raw low-order reconstruction for the canonical catalogue
```

The generator consumes the original orthogonal coefficient transcript through
`OrthogonalInteractionPrecision`; reconstructed diagonal variances are not the
optimization objective.

Generator:

```text
precision_operator: orthogonal_interaction
objective: exact released weighted quadratic coefficient likelihood
candidate profile: existing frozen QDTE Standard
candidates per iteration: 4,096
iterations: exactly 5,000
transport: atom_flow / batch
confidence stop: disabled
P3: disabled
shrinkage: disabled
entropy: disabled
structured cycles: disabled
fission: disabled
true utility during generation: disabled
```

Each cell uses exactly one 5000-step generation stage. No warm-start from a
different method or previous epsilon is allowed.

## 4. Frozen Matrix

```text
datasets:
  adult_sage_strong
  acs_sage_strong
  br2000_sage_strong
  nltcs_sage_strong

epsilon:
  0.1
  0.3

seeds:
  0
  1
  2

total cells: 24
```

Official AIM references already exist for all 24 cells and must be reused.
They are not rerun after seeing WP9 outputs.

## 5. Frozen Randomness

For integer protocol seed `s`, independent deterministic SeedSequence domains
are used:

```text
measurement seed = SeedSequence([s, 0xB453])
generation seed  = SeedSequence([s, 0x57A63, 0])
```

The domain constants match the already sealed WP8a full-Static mechanism, but
WP9 writes a new namespace and its own complete provenance. Existing outputs
are never overwritten.

## 6. Blind Boundary

Generation may load the private table only in the Gaussian measurement
mechanism. QDTE receives only:

```text
released orthogonal coefficients
released variances / rho metadata
public schema
public n
```

During generation, the following are forbidden:

```text
true query answers
true MAE/RMSE/TVD/MaxError
truth-based stopping
truth-based candidate generation or scoring
truth-based hyperparameter selection
```

Each cell records `true_utility_evaluated=false`. The evaluator cannot run
until all 24 cells are complete, their hashes are sealed, and the panel writes:

```text
offline_evaluation_authorized=true
```

The evaluator source hash is part of the panel plan before the first run.

## 7. Required Mechanism Checks

Every cell must prove:

```text
privacy.mode == dp
adjacency == add_remove
rho_spent == rho_declared within tolerance
epsilon is recomputed from actual rho_spent
every strategy block has positive finite variance
every public pair satisfies the 20,000-cell cap
the generation process reused the sealed measurement transcript byte-for-byte
precision_operator == orthogonal_interaction
score_backend == precision_operator
incremental answer drift audits pass
no true utility field is emitted
5,000 iterations complete
20,480,000 candidates are requested and scored
```

Any failed cell invalidates the panel. Partial output requires a new output
root or explicit archival; it cannot be silently resumed through changed code.

## 8. Evaluation

Primary metrics:

```text
MAE
RMSE
AvgTVD
```

Tail metrics, reported separately:

```text
MaxTVD
MaxError
```

For dataset `d`, epsilon `e`, seed `s`, define the primary ratio:

```text
R[d,e,s] = geometric_mean_m(
    Static-ICE-Exact metric[m] / Official-AIM metric[m]
)
```

The dataset-epsilon cell ratio is the geometric mean over the three seeds.
The dataset aggregate is the geometric mean over both epsilon values, all
three seeds, and all three primary metrics.

The overall confidence interval uses a deterministic hierarchical paired
bootstrap with 20,000 replicates:

1. resample the four datasets with replacement;
2. within each sampled dataset, resample its two epsilon cells with
   replacement;
3. within each sampled dataset-epsilon cell, resample the three paired seeds
   with replacement;
4. aggregate paired log ratios and exponentiate;
5. report the 2.5% and 97.5% quantiles.

Bootstrap RNG seed:

```text
0x1CE20260715
```

## 9. Frozen Promotion Gate

WP9 confirms the low-budget Static-ICE candidate only if all conditions hold:

```text
cell wins:
  at least 6 of 8 dataset-epsilon cell ratios are < 1

overall paired evidence:
  overall primary ratio < 1
  hierarchical bootstrap 95% upper bound < 1

dataset safety:
  every dataset primary aggregate <= 1.05

tail safety:
  every dataset aggregate MaxTVD ratio <= 1.15
  every dataset aggregate MaxError ratio <= 1.15
```

Decision labels:

```text
confirm_low_budget_static_ice
retain_static_ice_as_development_candidate
```

Failure does not invalidate QDTE or the pure-interaction measurement theorem.
It means this exact frozen end-to-end profile lacks sufficient multi-seed
evidence for a broad low-budget AIM-dominance claim.

## 10. Forbidden Post-Hoc Actions

After any WP9 utility is read, do not:

- switch exact precision to the diagonal adapter in failed cells;
- change the 20,000-cell cap;
- increase candidates or iterations only on losing datasets;
- add P3, BootDiag, shrinkage, entropy, cycles, fission, or private selection;
- drop Adult or BR2000 from the aggregate;
- relax 6/8 wins, dataset safety, tail safety, or bootstrap requirements;
- choose a sparse PGM graph from true errors;
- reuse WP9 cells to promote a newly designed variant.

Any later method is a new protocol and must validate first on holdout evidence.

## 11. Artifacts

Each cell must store:

```text
schema.json
queries.json
measurements.json
config_resolved.yaml
synthetic_initial_encoded.npy
synthetic_encoded.npy
run_status.json
runtime.json
mechanism_manifest.json
```

The panel stores:

```text
panel_plan.json
panel_status.json
sealed_panel_manifest.json
```

The offline evaluator stores:

```text
metrics.csv
per-cell raw evaluator JSON
gate_summary.json
```
