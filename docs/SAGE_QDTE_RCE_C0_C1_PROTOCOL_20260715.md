# SAGE-QDTE-RCE C0/C1 Frozen Protocol

Date: 2026-07-15

## 0. Identity and scope

```text
protocol_id: SAGE-QDTE-RCE-C0-C1-20260715-v3
method_id: SAGE-QDTE-RCE-v1
adjacency: unbounded / add_remove
status: method-limit research track
```

This protocol implements the expert-selected RCE route without modifying any
frozen QDTE Base, Static-ICE, WP9, WP10a, or WP10b artifact. The first causal
change is limited to the estimator/generator objective:

```text
control:   point-target Static-ICE QDTE
candidate: row-realizable confidence-set minimum-KL QDTE
```

The following remain disabled in C1:

```text
support coarsening
additional measurements or rho reallocation
confidence-dual adaptive refinement
Gaussian fission
new structured-cycle profile
P3 / BootDiag / shrinkage
private query selection
true-utility stopping or tuning
```

## 1. Frozen statistical object

The confidence operator uses the independent raw Helmert coefficient
coordinates of the existing hierarchical interaction transcript. It does not
treat correlated reconstructed cells as independent observations.

For coefficient residual `r = z - A(D_syn)` and raw coefficient variances
`v`, freeze:

```text
alpha_total = 0.05
alpha_l2    = 0.025
alpha_linf  = 0.025
rank        = number of positive-variance identifiable coefficients
m           = number of coordinate-tube coefficients
c2          = chi2.ppf(1 - alpha_l2, rank)
c_inf       = norm.ppf(1 - alpha_linf / (2*m))
```

The feasible set is:

```text
sum_i r_i^2 / v_i <= c2
max_i |r_i| / sqrt(v_i) <= c_inf
```

Zero-variance/null coordinates are excluded from both norms and must be
reported explicitly. No family-specific alpha split or weight is allowed.

The prior is the released complete one-way product distribution with
Dirichlet/Laplace pseudocount exactly `1`. Released one-way cell counts are
first projected to their public simplex. The row count `n` and schema are
explicitly public.

The implementation is in rate space, matching the estimator definition:

```text
rce_regularizer = D_KL(p_empirical || p0)
                = count_scaled_regularizer / public_n
rce_edit_cost   = configured_lambda_cost * edit_cost / public_n
```

The `/ public_n` edit-cost normalization preserves the declared cost as the
same per-record algorithmic tie-break while preventing it from changing scale
when the public row count changes. The confidence constraints are already
dimensionless standardized quantities and are not divided by `n` again.

The earlier unsealed development plan with protocol suffix `v1` was stopped
before any offline true-utility evaluation after an audit found that it reused
the count-scaled `n * D_KL` state with O(1) confidence dual updates. No v1 cell
was sealed or admitted as evidence. Version `v2` fixed this mathematical scale
and introduced core-implementation hash sealing, which `v3` retains.

The unsealed `v2` plan verified the corrected search trajectory but was also
stopped before offline evaluation because the integer-incumbent JSON diagnostic
divided an already normalized `D_KL` value by `n` a second time. This did not
affect scoring, transport, incumbent selection, or generated rows, but `v3`
fixes the certificate field and hard-gates that `best_kl_per_row` equals the
stored normalized regularizer. Neither earlier plan contributes result rows.

## 2. Lexicographic integer objective

The integer table objective is frozen in two stages:

1. minimize the maximum nonnegative standardized confidence violation `s`;
2. among tables attaining the best observed/certified `s`, minimize
   `KL(p || p0)`.

Confidence thresholds may not be relaxed after seeing utility. A positive
`s_star` is an infeasibility/optimizer certificate, not permission to change
alpha.

RCE candidate and batch scores must be exact finite differences of the active
Lagrangian terms:

```text
ellipsoid quadratic term
coordinate-tube linear term
full-row empirical KL term
declared edit cost
```

Accepted batches are recomputed in float64 audit mode. The existing QDTE
residual convention remains unchanged in query space:

```text
residual[q] = target_projected[q] - answer_syn[q]
```

## 3. Relaxed ceiling terminology

For a small domain whose complete row atom set can be enumerated, the project
must solve the exact convex relaxed RCE problem and report its unique optimum.

For Adult/BR2000, the full row domain and dense pairwise exponential-family
partition function are not tractable by enumeration. Until a globally
certified column/MAP oracle exists, any convex solve over a discovered atom
set must be named:

```text
restricted_relaxed_rce
```

It is an upper bound on the globally relaxed minimum-KL objective over the
same confidence constraints, not a proof of the global statistical ceiling.
The artifact must report:

```text
atom support construction
support size and hash
primal feasibility
restricted optimum
best available column/reduced-cost certificate
whether global optimality is certified
```

No paper claim may call a restricted optimum the global relaxed RCE ceiling.

## 4. C0 ceiling decomposition

```text
datasets: Adult, BR2000
epsilon: 0.1, 0.3
seeds: 0, 1, 2
generator budget: 5000 iterations x 4096 candidates
```

The four diagnostic cells are:

| Support/coarsening | Interaction information | Role |
| --- | --- | --- |
| released-only | frozen DP transcript | releasable mechanism |
| oracle | frozen DP transcript | support/coarsening diagnostic only |
| released-only | clean strategy coefficients | measurement-noise diagnostic only |
| oracle | clean strategy coefficients | joint method-class diagnostic only |

Oracle artifacts are offline diagnostics. They cannot enter generation,
selection, hyperparameters, or paper-method promotion.

Before the Adult/BR2000 oracle cells run, the runner must freeze exact support
definitions and prove that no oracle-derived object is reachable from the
released-only arm.

## 5. C1 same-transcript causal panel

For every dataset-epsilon-seed cell, the control and candidate reuse the exact
same sealed WP9 transcript, schema, public `n`, initialization seed, generation
seed, candidate settings, iteration count, and evaluator. The candidate may
consume the transcript only through the RCE confidence operator and released
one-way prior.

Required generation diagnostics:

```text
minimum confidence slack s_star
ellipsoid value / threshold
maximum tube statistic / threshold
primal feasibility
dual nonnegativity
complementarity residual
primal-dual/restricted-edit gap
KL to released one-way prior
active ellipsoid/tube constraints
accepted single edits and cycle edits
restricted relaxed optimum and integer gap, when available
global relaxed certificate status
```

True metrics are computed only after all twelve candidate artifacts are sealed.
No seed-0 decision is allowed.

## 6. C1 interpretation

C1 does not use a one-cell promotion stop. It classifies the mechanism:

```text
RCE improves point-target QDTE:
  estimator regularization has causal value; continue ceiling analysis.

restricted relaxed improves but integer RCE does not:
  row-realization / optimizer gap; strengthen the declared row oracle.

clean interactions improve but DP interactions do not:
  measurement information bottleneck; C3 or low-rank measurement may be needed.

released RCE has benefit but high-cardinality noise remains dominant:
  C2 support coarsening may be authorized.

oracle support + clean interactions + certified relaxed RCE still lose:
  current confidence-set product-prior model class lacks ceiling.
```

## 7. DP and reproducibility boundary

In `privacy.mode=dp`, no RCE module may import, receive, or load real rows,
exact answers, offline metrics, or oracle support. Released-only generation is
a separate process whose input surface is limited to:

```text
sealed measurement artifact
public schema
public row count
public config
generation RNG seed
```

Every run saves source hashes, config, transcript hash, initial-table hash,
candidate/iteration counts, certificates, metrics separation flags, and logs.

## 8. First implementation gate

Before any formal C0/C1 generation, all of the following must pass:

```text
nominal simulated confidence coverage
small-domain relaxed uniqueness
KL single-edit exactness
quadratic + tube + KL batch exactness
singular/zero-precision support handling
primal/dual certificate tests
DP no-truth reachability test
Base/Static-ICE regression suite
```

Only after this gate may the twelve same-transcript C1 candidate runs begin.
