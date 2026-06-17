# Consistency Projection Expert Note

Date: 2026-06-11

## 1. Project Background

We are building QDTE, a differentially private synthetic tabular data generator.

The current pipeline is:

1. Build a workload of counting queries over the real table.
2. Measure those queries once with DP Gaussian noise.
3. Optionally post-process the noisy measurements into projected targets.
4. Initialize a synthetic table.
5. Evolve the synthetic table through local row edits.
6. Score edits only against noisy/projected measurements.

The QDTE optimization invariant is:

```text
residual[q] = target_projected[q] - answer_syn[q]
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
```

All active optimization must use only noisy/projected measurements, query definitions, variances, and synthetic data. Exact true answers are allowed only for offline evaluation.

## 2. Why Consistency Projection Matters

The candidate-generation issue we observed is that a local row edit may help one query while hurting many overlapping queries. For example, a `single_query` repair can be locally correct for its target query, but globally negative under the full workload loss.

In a 2000-step smoke experiment, the candidate diagnostics were:

| Variant | Final measured loss | Offline true RMSE | Mean target component | Mean collateral component | Target-positive but full-negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_query` | 66.653 | 0.003801 | +0.2717 | -0.6862 | 97.31% |
| `masked_single_query` | 65.902 | 0.003811 | +0.2879 | -0.6739 | 97.14% |
| `residual_weighted_mutation` | 63.991 | 0.003817 | -0.0033 | -0.4581 | 4.10% |

This suggests that noisy/projected residuals are not sufficiently consistent across overlapping queries. We therefore want a principled consistency projection before QDTE optimization.

## 3. What Went Wrong In The Current Consistency Projection

The repository currently contains a `local_marginal_ipf` consistency projection. It does:

1. Fit one local table for each query scope.
2. Compute shared marginals between overlapping scopes.
3. Average those shared marginals using a scope-level precision.
4. Scale local tables to match the averaged marginals.
5. Repeat this reconciliation.

This is not the desired global weighted projection objective. It can make local tables mutually consistent while drifting away from the original noisy measurements.

Smoke target-vs-true audit:

| Target version | Target-vs-true RMSE |
| --- | ---: |
| Raw noisy target | 0.006508 |
| Ordinary simplex/clip projection | 0.006266 |
| Current full consistency projection | 0.013520 |
| Current consistency projection with preprojected input | 0.013522 |
| Per-scope weighted LSQ without cross-scope IPF | 0.004891 |

The likely culprit is the cross-scope averaging/IPF step, not the idea of local least-squares fitting itself.

## 4. Desired Statistical Target

Let:

```text
theta = exact true query answer vector, unknown during optimization
y     = noisy measured query answer vector
eta   = DP noise
y = theta + eta
E[eta] = 0
Sigma = covariance of eta
W = Sigma^{-1}, or diagonal inverse variances if measurements are independent
```

We want a DP post-processing estimator:

```text
theta_hat = projection(y, known public metadata)
```

The first corrected version should be **unbiased**, so we should avoid nonnegativity constraints initially. The clean baseline is an equality-only weighted projection:

```text
min_z  0.5 * (z - y)^T W (z - y)
s.t.   A z = b
```

where `A z = b` encodes consistency constraints known to be satisfied by the true answer vector.

The closed form is:

```text
z_hat = y - W^{-1} A^T (A W^{-1} A^T)^+ (A y - b)
```

where `^+` denotes a pseudoinverse if constraints are redundant.

If:

```text
E[y] = theta
A theta = b
```

then:

```text
E[z_hat] = theta
```

because this is an affine projection onto a constraint set that contains the true answer vector. Thus equality-only consistency projection should preserve unbiasedness under zero-mean noise.

## 5. Known Dataset Size N

We know the real dataset size `N`. In our setting, `N` is treated as public/non-private metadata.

For any complete partition of the domain, its cell counts must sum to `N`. This is an affine equality constraint:

```text
sum_i z[cell_i] = N
```

This should be included directly in `A z = b`, with right-hand side `N`.

Examples:

```text
oneway(A=0) + oneway(A=1) + ... + oneway(A=K-1) = N
```

For a two-way complete partition:

```text
sum_a sum_b twoway(A=a, B=b) = N
```

For nested prefix counts over a numeric-binned column:

```text
prefix(X <= max_bin) = N
```

For consistency between scopes:

```text
oneway(A=a) - sum_b twoway(A=a, B=b) = 0
oneway(B=b) - sum_a twoway(A=a, B=b) = 0
twoway(A=a, B=b) - sum_c threeway(A=a, B=b, C=c) = 0
```

For range/prefix constraints:

```text
range(X in [l, r]) - prefix(X <= r) + prefix(X <= l-1) = 0
```

The important point: `N` should not be a separate heuristic pre-step. In the corrected projection, the `N` constraints and cross-scope consistency constraints should be solved **jointly** in the same weighted projection.

If we use an iterative optimizer such as ADMM, the `N` constraints should be present from the beginning and maintained as part of the same global objective. Adding `N` before a heuristic cross-scope averaging step is not enough; the iteration must remain anchored to the original noisy measurements.

## 6. Full Joint Table Limitation

Exact full global consistency over all attributes normally requires a full joint table. This is infeasible for realistic tabular domains because the number of cells is:

```text
prod_j cardinality(attribute_j)
```

Therefore the practical goal is not full global joint projection. The realistic goal is **local-scope consistency projection**.

If a measured query itself spans a huge scope, then even a local table for that query may be infeasible. Options:

1. Exclude high-scope queries from consistency projection.
2. Keep high-scope queries as standalone noisy measurements with no strong consistency constraints.
3. Add only weak constraints such as upper/lower marginal bounds, if using inequality constraints later.
4. Use approximate structured representations such as sparse support, factorized tables, or decision diagrams.
5. Move high-scope queries to held-out evaluation instead of active measured workload.

For the first unbiased version, we should only include equality constraints that are exact, tractable, and clearly satisfied by the true query vector.

## 7. Proposed Implementation Plan

### Stage 1: Equality-Only Query-Space Projection

Implement:

```text
min_z 0.5 * (z - y)^T W (z - y)
s.t.  A z = b
```

This stage does not introduce nonnegativity or simplex inequality constraints.

Candidate constraints:

1. Complete partition sums equal `N`.
2. Lower-order marginal equals sum of higher-order marginal cells.
3. Prefix/range linear identities.
4. Duplicate query identities, if any remain after query construction.

Expected property:

- unbiased under zero-mean noise;
- lower variance in the constrained directions;
- easy to audit by checking `A z_hat - b`.

Implementation notes:

- Build sparse `A`.
- Use diagonal `W = diag(inv_variance)` initially.
- Solve with sparse linear algebra:

```text
M = A W^{-1} A^T
lambda = M^+ (A y - b)
z_hat = y - W^{-1} A^T lambda
```

- Handle redundant constraints with least-squares or pseudoinverse.

### Stage 2: Local-Table Equality Projection

If query-space constraints are insufficient, build local scope tables without nonnegativity:

```text
variables:
  T_s for selected local scopes s

objective:
  min 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q

constraints:
  sum(T_s) = N
  shared marginals of overlapping scopes are equal
```

This remains an equality-constrained weighted least-squares problem and can remain unbiased if the true local marginals satisfy the constraints.

### Stage 3: Optional Inequality-Constrained Version

Only after the unbiased baseline is understood, consider:

```text
T_s >= 0
```

or simplex projection.

This can reduce MSE but may introduce finite-sample bias. It should be treated as a separate estimator, not mixed into the unbiased baseline.

### Stage 4: ADMM For Scale

ADMM should be used only as an optimizer for the same corrected objective, not as a new heuristic.

Each local scope update should keep the original noisy measurements in its local objective. Shared marginals should be coupled with consensus variables and dual variables.

If ADMM converges to the equality-constrained objective, it has the same statistical target as the direct solver. Early stopping or a surrogate objective can introduce optimization bias.

## 8. Open Questions For Expert Discussion

1. Is the equality-only query-space projection sufficient for our workload families, or do we need local table variables?
2. Which exact linear consistency constraints should we include first?
3. How should we handle high-scope queries whose exact local tables are infeasible?
4. Should `N` be treated as public and exact in all experiments?
5. With diagonal Gaussian measurements, is the closed-form weighted projection above the right unbiased estimator?
6. If measurement groups have correlated noise or post-processing creates covariance, how should `W` be represented?
7. Is it acceptable to keep original diagonal variances after equality projection, or should we propagate projected covariance?
8. Should nonnegativity be avoided entirely for the main method, or reported as a biased but possibly lower-MSE variant?
9. For complex scope hypergraphs, is sparse direct solve enough, or should we move directly to ADMM/junction-tree-style decomposition?
10. Is there any nontrivial estimator that always outputs nonnegative counts, respects known total `N`, and remains exactly unbiased for every true nonnegative count vector under two-sided Gaussian/discrete-Gaussian noise?
11. If exact unbiasedness and nonnegativity are incompatible, what is the right target for the paper:
    - equality-only unbiased projection;
    - nonnegative biased projection with lower MSE;
    - or a two-stage method with unbiased targets for accounting/analysis and feasible targets for optimization?

## 9. Immediate Recommendation

Do not use the current `local_marginal_ipf` projection as the main QDTE preprocessing layer.

The next implementation should be:

1. Build equality constraints `A z = b` for known-safe consistency relations.
2. Include known dataset size `N` as right-hand-side constraints for complete partitions.
3. Solve the equality-only weighted projection.
4. Compare target-vs-true offline RMSE against:
   - raw noisy target;
   - ordinary simplex/clip projection;
   - per-scope LSQ without cross-scope reconciliation;
   - current `local_marginal_ipf`.
5. Only after this baseline is verified, consider nonnegative local-table QP or ADMM.

## 10. Implementation Status

Implemented after this note was drafted:

- Added `projection.consistency.method: query_space_lsq`.
- The implemented estimator solves the equality-only weighted projection in query-answer space:

```text
min_z 0.5 * (z - y)^T W (z - y)
s.t.  A z = b
```

- It constructs sparse equality constraints from existing measured queries only:
  - complete EQ cell partitions sum to known public `N`;
  - queries are tied to complete cell partitions on the same or a superset scope when the required cell queries already exist;
  - no full joint table is materialized;
  - no nonnegativity, clipping, or simplex inequality projection is applied.
- It uses `scipy.sparse.linalg.lsmr` on the weighted normal equations:

```text
A W^{-1} A^T lambda = A y - b
z = y - W^{-1} A^T lambda
```

Initial smoke audit:

| Target version | Target-vs-true RMSE |
| --- | ---: |
| Raw noisy target | 0.006508 |
| Ordinary simplex/clip projection | 0.006266 |
| Old `local_marginal_ipf` | 0.013520 |
| New `query_space_lsq` | 0.004981 |

Projection diagnostics on the smoke workload:

- complete cell partitions: `10`
- equality constraints: `170`
- constraint nonzeros: `1974`
- initial max constraint violation: `33.90`
- final max constraint violation: `6.52e-7`
- min projected answer: `-17.58`
- max projected answer: `1000.00`

The negative projected answer is expected for the unbiased equality-only version. It is not clipped. This means the projected target can be outside the feasible nonnegative query-answer cone of any synthetic dataset, so the measured objective may have a nonzero floor. That is an acceptable tradeoff for the unbiased baseline and should be compared separately against biased/nonnegative variants.

## 11. Nonnegativity And Unbiasedness

The equality-only projection can output negative counts because the DP noise is two-sided and the projection only enforces affine equalities. If a true count is near zero and the noise is sufficiently negative, an affine unbiased estimator can remain negative.

Requiring the estimator to always output nonnegative counts is generally incompatible with exact unbiasedness for all possible true counts under two-sided additive noise. A simple one-dimensional intuition:

```text
y = theta + noise
theta >= 0
E[noise] = 0
```

If an estimator `g(y)` must satisfy `g(y) >= 0` for all outputs and also be unbiased at `theta = 0`, then:

```text
E[g(noise)] = 0
```

Since `g(noise) >= 0`, this forces `g(noise) = 0` almost surely under the noise distribution at `theta=0`. For Gaussian or discrete Gaussian noise with overlapping support across different `theta`, this leaves no room for the estimator to be unbiased for positive `theta` as well. In other words, exact unbiasedness and guaranteed nonnegativity usually cannot both hold except in trivial or restricted cases.

This does not mean nonnegative projection is useless. It means it should be presented as a biased or shrinkage estimator that can reduce MSE and improve optimization feasibility. For QDTE we may need to compare two targets:

1. `query_space_lsq`: equality-only, affine, unbiased baseline.
2. A future feasible projection: nonnegative/local-table constrained, likely biased, but better aligned with the synthetic table optimization geometry.

## 12. Literature Notes On Nonnegative / Consistent DP Post-Processing

This is a recognized issue in the DP literature, especially for histograms, hierarchical census products, and private synthetic data from marginal measurements.

Key references:

1. Hay, Rastogi, Miklau, and Suciu, "Boosting the Accuracy of Differentially-Private Histograms Through Consistency" (arXiv:0904.0942).
   - Early and directly relevant work.
   - Uses consistency constraints in post-processing after noisy histogram measurement.
   - The goal is a final output that is DP by post-processing, consistent, and more accurate.
   - This supports the idea that consistency projection should be discussed as a standard DP accuracy tool.
2. Zhu, Van Hentenryck, and Fioretto, "Bias and Variance of Post-processing in Differential Privacy" (arXiv:2010.04327).
   - Directly relevant to our nonnegativity concern.
   - Studies projecting DP outputs onto feasible/domain-constrained regions.
   - Explicitly frames the issue that post-processing may introduce bias and change variance.
   - This is likely the most important citation for our "nonnegative projection is useful but may be biased" discussion.
3. Fioretto, Van Hentenryck, and Zhu, "Differential Privacy of Hierarchical Census Data: An Optimization Approach" (arXiv:2006.15673).
   - Treats hierarchical counts that must be consistent across levels.
   - Uses optimization to redistribute DP noise under consistency constraints.
   - Relevant for scalable constrained post-processing with known totals and hierarchical constraints.
4. Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (arXiv:2204.08986).
   - Real deployment example.
   - Uses zCDP measurements, public/policy invariants, and post-processing to produce a microdata file.
   - Important precedent for using structural constraints and invariants after DP measurement.
5. McKenna, Sheldon, and Miklau, "Graphical-model based estimation and inference for differential privacy" (arXiv:1901.09136).
   - Estimates high-dimensional distributions from noisy low-dimensional marginals.
   - Relevant to local-scope feasible/nonnegative projection when full joint tables are infeasible.
6. McKenna, Miklau, and Sheldon, "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data" (arXiv:2108.04978).
   - Describes Private-PGM as post-processing noisy marginals to estimate a distribution.
   - Strong precedent for "measure marginals, then fit a nonnegative distribution/synthetic data target".
7. McKenna, Pradhan, Sheldon, and Miklau, "Relaxed Marginal Consistency for Differentially Private Query Answering" (arXiv:2109.06153).
   - Addresses scalability limits of exact marginal consistency.
   - Relevant because our workload can form complex hypergraphs; relaxed consistency may be necessary at Adult scale.
8. Ghazi, Kamath, Kumar, Manurangsi, and Sealfon, "Denoising the US Census: Succinct Block Hierarchical Regression" (arXiv:2603.10099).
   - Recent Census post-processing work.
   - Separates statistically optimal linear unbiased regression from additional structural constraints.
   - This matches our proposed split between equality-only unbiased projection and feasible/nonnegative projection.

Implication for QDTE:

- Our paper should explicitly separate two objectives:
  - statistical target quality: equality-only weighted projection can be unbiased and variance-reducing;
  - optimization feasibility: nonnegative/local-table projection can make the residual field more coherent for synthetic table optimization, but generally introduces bias.
- This separation is not a weakness. It is consistent with the literature: constrained post-processing is standard, useful, and DP-safe, but its statistical effects should be measured.
- A worthwhile next contribution is to compare:
  - raw noisy measurements;
  - equality-only `query_space_lsq`;
  - nonnegative feasible projection;
  - relaxed marginal consistency / graphical-model projection;
  - and their effect on QDTE edit convergence and true-query utility.

## 13. Feasible Query-Space Projection Implementation

Implemented after expert discussion:

- Added `projection.consistency.method: query_space_feasible_lsq`.
- It solves the biased feasible query-space projection:

```text
min_z 0.5 * (z - y)^T W (z - y)
s.t.  A z = b
      0 <= z <= N
```

- It uses the same sparse equality constraints as `query_space_lsq`.
- To make SLSQP stable, it removes redundant equality rows through rank-revealing QR before solving.
- This is not a full joint-table feasibility projection. It is a nonnegative, bounded, equality-consistent query-vector projection over the measured query coordinates.

Smoke target audit:

| Target version | Target-vs-true RMSE | Min | Max |
| --- | ---: | ---: | ---: |
| Raw noisy target | 0.006508 | -17.58 | 1002.53 |
| Ordinary simplex/clip projection | 0.006266 | 0.00 | 1000.00 |
| Old `local_marginal_ipf` | 0.013520 | 0.00 | 1000.00 |
| Equality-only `query_space_lsq` | 0.004981 | -17.58 | 1000.00 |
| Feasible `query_space_feasible_lsq` | 0.004724 | 0.00 | 1000.00 |

Smoke feasible projection diagnostics:

- equality constraints: `170`
- independent constraints after QR: `98`
- redundant constraints: `72`
- final max constraint violation: `1.71e-13`
- active lower-bound answers: `8`
- active upper-bound answers: `2`

2000-step QDTE smoke results under `query_space_feasible_lsq`:

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `enumerated_local` | 18.865119 | 0.00243210 | 0.00383776 | 1942 |
| `residual_weighted_mutation` | 19.382165 | 0.00238272 | 0.00377505 | 1757 |
| `random_mutation` | 20.662170 | 0.00244033 | 0.00391000 | 1790 |
| `single_query` | 20.797150 | 0.00246502 | 0.00391420 | 1573 |
| `masked_single_query` | 21.426172 | 0.00253086 | 0.00400360 | 1459 |

Interpretation:

- The feasible projection supports the expert's claim at the target level on this smoke run: target-vs-true RMSE is lower than equality-only `query_space_lsq`.
- It also lowers QDTE measured loss compared with equality-only `query_space_lsq`.
- However, single-seed final synthetic true RMSE is not uniformly better than equality-only projection. This may be optimization/candidate-strategy interaction, seed noise, or the fact that query-space feasibility is still weaker than local-table/global feasibility.
- This should be evaluated with multi-seed runs and residual-field diagnostics before becoming the default.

## 14. Local-Table Feasible Projection Implementation

Implemented the stronger local-table feasible projection:

- Config method: `projection.consistency.method: local_table_feasible_lsq`.
- Variables are latent nonnegative local tables for every measured query scope.
- The objective stays anchored to the original noisy measurements:

```text
min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
s.t.  T_s >= 0
      sum(T_s) = N
      shared marginals of overlapping scopes agree
```

This is intentionally different from the old `local_marginal_ipf`:

- old IPF first fits local tables and then repeatedly averages shared marginals;
- new local-table feasible projection solves one weighted constrained objective against the original noisy answers;
- both are biased because of nonnegativity, but the new version has a clearer objective and auditable constraints.

Smoke target audit with the same noisy measurements:

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

2000-step QDTE smoke results under `local_table_feasible_lsq`:

| Variant | Final measured loss | Final true MAE | Final true RMSE | Accepted edits |
| --- | ---: | ---: | ---: | ---: |
| `single_query` | 5.822277 | 0.002481 | 0.003879 | 1461 |
| `enumerated_local` | 6.616903 | 0.002556 | 0.003985 | 1820 |
| `masked_single_query` | 6.811439 | 0.002490 | 0.003934 | 1423 |
| `random_mutation` | 7.147311 | 0.002498 | 0.003973 | 1756 |
| `residual_weighted_mutation` | 8.096271 | 0.002584 | 0.004143 | 1899 |

Interpretation:

- The new projection reaches the low measured-loss regime that motivated the old IPF investigation, but with much better target-vs-true RMSE than old IPF.
- The result supports the hypothesis that query-space feasible projection was too weak because it did not impose latent local-table feasibility on mixed scopes lacking complete measured cell partitions.
- `single_query` now wins measured loss on this single seed, so local-table feasibility does help directional edits. However `masked_single_query` still does not beat `single_query`; this suggests the current mask proposal is still losing too much useful candidate support or over-constraining edits.
- Candidate diagnostics still show high target-positive/full-negative rates for single-query-style proposals, so local-table target consistency does not by itself eliminate all collateral conflicts. This may be because one row edit affects many scopes, while local-table feasibility only makes the measurement target coherent, not every local edit direction globally aligned.
- Runtime is materially higher than query-space projection because each run performs SLSQP over local table cells and dense rank-revealing QR over the equality constraints. This is fine for smoke, but Adult-scale use likely needs ADMM/PGM-style optimization.

## 15. JAX Active-Set Local-Table Projection

Added an optional GPU-oriented method:

```text
projection.consistency.method: local_table_feasible_jax
```

This method is additive; the CPU SLSQP reference `local_table_feasible_lsq` is still available.

The first JAX version used a penalty approximation. It has now been replaced by a dense active-set QP solver that targets the same hard-constrained objective as CPU SLSQP:

```text
min_T 0.5 * sum_q (a_q^T T_scope(q) - y_q)^2 / var_q
s.t.  T_s >= 0
      sum(T_s) = N
      shared marginals agree across overlapping scopes
```

Implementation summary:

- build the same local-table measurement matrix and equality constraint matrix;
- remove redundant equality rows with rank-revealing QR;
- maintain an active set of local-table cells fixed at zero;
- solve dense equality-constrained KKT least-squares systems with JAX float64;
- add blocking variables when a candidate step would make a table cell negative;
- release active variables when lower-bound multipliers violate KKT conditions.

Smoke audit:

| Method | Projection time | Target-vs-true RMSE | Weighted objective | Final max violation |
| --- | ---: | ---: | ---: | ---: |
| CPU `local_table_feasible_lsq` | about 100s in this environment | 0.00334076 | 67.878562163 | 1.42e-13 |
| JAX `local_table_feasible_jax` active set | 35.7s audit / 36.3s full run | 0.00334075 | 67.878562161 | 8.64e-12 |

The JAX active-set solver now matches CPU SLSQP target quality and hard-constraint accuracy on the smoke workload. It is still a dense method, so Adult-scale use may need sparse KKT, ADMM, or a PGM-style solver.
