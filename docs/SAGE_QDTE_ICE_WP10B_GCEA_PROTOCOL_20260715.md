# SAGE-QDTE-ICE WP10b GCEA Protocol

Date: 2026-07-15

Protocol ID:

```text
SAGE-QDTE-ICE-WP10B-GCEA-20260715-v1
```

Candidate ID:

```text
SAGE-QDTE-Static-ICE-GCEA-v1
```

## 1. Frozen decision

WP10b is the only authorized final general-cardinality ICE mechanism.

```text
decision: Q1-B
authorized change: rho allocation vector only
fallback: any public or end-to-end gate failure => Q1-A
```

No sparse tree, family weight, dimension exponent, shrinkage, P3/covariance
upgrade, entropy term, structured-cycle upgrade, private selector, coverage
refinement, stopping change, or truth-guided tail guard may be added.

## 2. Public-only inputs

GCEA may read only:

```text
public schema/cardinalities
public evaluator queries and partition groups
public strategy matrices and reconstruction maps
public sensitivities
public n
declared total rho
frozen control allocation rho0
```

It may not read the private table, true answers, released noisy transcript,
offline utility, or any true/released residual ranking.

Let `Q_R` be the evaluator queries exactly reconstructed by the frozen
orthogonal strategy. Let `B_R` contain only complete evaluator partition
groups whose every coordinate is in `Q_R`. Unsupported high-order queries are
excluded from the public risk objective but remain in the final evaluator.
Any strategy block with no coefficient in `Q_R` is fixed at its control rho;
all remaining blocks share `rho_movable`.

For query `q` and measurement block `g`, define the public unit-rho variance
coefficient

```text
a_qg = r_qg^T C_g r_qg
C_g  = 0.5 * Delta_g^2 * I
```

and

```text
v_q(rho) = v_q_fixed + sum_g a_qg / rho_g
s_q(rho) = sqrt(v_q(rho)).
```

The implementation must verify every included reconstruction to maximum
absolute residual `<= 1e-10` using the public finite domain.

## 3. Frozen evaluator risks

All risks use count-space covariance and divide by public `n` to report rate
units.

```text
R_MAE = sqrt(2/pi) / (n * |Q_R|) * sum_q s_q

R_RMSE = 1 / (n * sqrt(|Q_R|)) * sqrt(sum_q v_q)

R_AvgTVD = sqrt(2/pi) / (2 * n * |B_R|)
           * sum_b sum_{q in b} s_q
```

`R_MAE` and `R_AvgTVD` are exact expectations for the raw zero-mean Gaussian
reconstruction. `R_RMSE` is the root-expected-MSE upper envelope.

Freeze:

```text
beta_tail = 0.05
c_beta = sqrt(2 * log(2 * |Q_R| / beta_tail))

U_MaxError = c_beta / n * max_q s_q

U_MaxTVD = c_beta / (2*n) * max_b sum_{q in b} s_q
```

The two tail quantities protect only the raw measurement reconstruction. They
are not claims about projected or final synthetic utility.

## 4. Frozen convex allocation

Let `rho0` be the frozen cell-reference `public_optimal` allocation. GCEA
solves

```text
minimize t

subject to
  R_MAE(rho)    <= t * R_MAE(rho0)
  R_RMSE(rho)   <= t * R_RMSE(rho0)
  R_AvgTVD(rho) <= t * R_AvgTVD(rho0)
  U_MaxError(rho) <= U_MaxError(rho0)
  U_MaxTVD(rho)   <= U_MaxTVD(rho0)
  sum_g rho_g = rho_movable
  rho_g > 0
```

The control is feasible, hence `t_star <= 1`. Each `s_q` is convex because it
is the nonnegative-orthant L2 composition of the convex coordinates
`rho_g^(-1/2)`. All five risk constraints are convex.

If stage one has multiple optima, stage two uniquely minimizes

```text
KL(rho / rho_movable || rho0 / rho_movable)
```

over the stage-one optimal face. Zero-control support, if any, remains fixed
and is excluded from KL.

## 5. Frozen numerical convention

The numerical implementation is part of the protocol and may not be changed
after reading public-gate output:

```text
variables: normalized positive movable shares p and epigraph t
solver: deterministic SciPy SLSQP
derivatives: analytic risk and constraint Jacobians
initial point: normalized control allocation
max iterations: 5000 per stage
solver ftol: 1e-13
numerical share floor: 1e-14
stage-two optimal-face tolerance: 1e-10 in dimensionless primary ratio
```

The floor is only a floating-point domain guard. A public gate passes only if
every final share is at least `1e6` times the floor, so the guard is inactive.

The reported KKT gap is the maximum of:

```text
equality residual
negative inequality slack
negative dual multiplier
complementarity residual
absolute Lagrangian stationarity residual
```

for both stages, using the dimensionless ratio formulation and SLSQP's
constraint multipliers. The final certificate reports the larger stage gap.

## 6. Public mechanism gate

Before generation, run only Adult and BR2000 at epsilon `0.1` and `0.3` from
public metadata. Both epsilon values must independently reproduce the same
normalized allocation up to numerical tolerance.

Every cell and both datasets must satisfy:

```text
t_star <= 0.97
MaxError envelope ratio <= 1.0000000001
MaxTVD envelope ratio <= 1.0000000001
relative sum-rho error <= 1e-12
allocation KKT gap <= 1e-8
reconstruction residual <= 1e-10
all shares finite, positive, and > 1e6 * numerical floor
```

Any failure produces:

```text
generation_authorized: false
decision: Q1-A
```

The threshold may not be relaxed and no alternative mechanism may be run.

## 7. One-time end-to-end panel, conditional on public gate

Only if the complete public gate passes:

```text
datasets: Adult, BR2000, ACS, NLTCS
epsilon: 0.1, 0.3
seeds: 10, 11, 12, 13, 14
adjacency: unbounded / add_remove
```

The candidate and frozen control use identical total rho, strategy blocks,
coefficient ordering, standard-normal draw per coefficient, public schema,
projection, initial table/hash, generation seed, exact precision operator,
5000 iterations, 4096 candidates per iteration, and evaluator. Only the rho
allocation differs.

## 8. Frozen promotion gate

Primary composite is the geometric mean over MAE, RMSE, and AvgTVD ratios.

General-cardinality versus control and AIM must each satisfy:

```text
aggregate primary <= 0.97
hierarchical-bootstrap 95% CI upper < 1
every Adult/BR2000 dataset-epsilon cell primary <= 1
every individual primary metric seed-GM ratio <= 1.05
```

Across all eight dataset-epsilon cells, candidate/control tail ratios must
satisfy:

```text
each MaxError and MaxTVD cell <= 1.15
GM MaxError ratio <= 1
GM MaxTVD ratio <= 1
```

Binary non-regression must satisfy:

```text
aggregate ACS/NLTCS primary versus control <= 1.02
each binary dataset-epsilon primary versus control <= 1.05
each binary dataset-epsilon primary versus AIM <= 1.00
```

Any failure freezes ICE as a binary/low-cardinality profile and returns the
paper's broad method to QDTE. No post-hoc rescue is authorized.
