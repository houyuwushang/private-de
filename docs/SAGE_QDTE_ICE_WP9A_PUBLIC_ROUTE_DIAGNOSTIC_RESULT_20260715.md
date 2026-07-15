# SAGE-QDTE-ICE WP9a Public-Route Diagnostic Result

Date: 2026-07-15

Protocol:

```text
SAGE-QDTE-ICE-WP9A-PUBLIC-ROUTE-DIAGNOSTIC-20260715-v1
```

## 1. Boundary and provenance

WP9a read only:

```text
public schema
public queries_full.json
public workload_groups.json
24 sealed WP9 DP measurement transcripts
```

It did not load exact rows, true answers, synthetic utility, AIM output, or an
offline evaluation cache. It generated no synthetic data and promoted no
method.

Formal artifacts:

```text
outputs/static_ice_wp9a_public_routes_formal_20260715/
  public_route_diagnostics.json
  allocation_risk_summary.csv
  released_tree_stability.csv
  manifest.json
```

The manifest records `true_data_loaded=false` and
`true_utility_evaluated=false`. All three output hashes were independently
verified after the run.

## 2. The implementation gap is now precise

The existing `public_optimal` allocator is correct for its declared objective,
but that objective is narrower than the full paper workload. It uses the
complete one-/two-way cell workload:

\[
c_a^{\mathrm{cell}}
=\sum_{q\in\text{one-/two-way cells}}\|L_{qa}\|_2^2.
\]

WP9a recomputed this quantity from public query predicates and exact
orthogonal reconstruction maps. Its maximum relative difference from every
existing `StrategyBlock.public_importance` was:

```text
ACS:     4.44e-16
NLTCS:   4.44e-16
BR2000:  1.98e-14
Adult:   6.91e-14
```

Thus the current formula and code are internally correct. The missing expert
step is deriving `c_a` from the actual public heterogeneous workload:

\[
c_a^{\mathrm{workload}}
=\sum_{q\in Q_{\le2}}w_q\|L_{qa}\|_2^2.
\]

## 3. Public workload coverage

| Dataset | Full queries | Exactly reconstructable from ICE order <=2 | Unsupported high-order |
| --- | ---: | ---: | ---: |
| ACS | 5,154 | 1,058 | 4,096 |
| NLTCS | 8,704 | 512 | 8,192 |
| BR2000 | 8,907 | 4,811 | 4,096 |
| Adult | 28,654 | 22,534 | 6,120 |

ACS, NLTCS, and BR2000 contain no extra reconstructable low-order scalar
queries beyond their one-/two-way cells. Adult additionally contains:

```text
mixed:                    5,558
prefix:                      87
range:                      668
low-order orthogonal mixed: 356
```

Adult therefore exposes a real mismatch between the allocation objective and
the declared evaluator workload.

## 4. Recoverable public allocation risk

The table reports

\[
R(c,\rho^*)/R(c,\rho^{\mathrm{WP9}}).
\]

Smaller values mean a lower analytically expected public measurement risk at
the same total rho. These are not final utility ratios.

| Dataset | Full reconstructable query L2 | Equal public-group L2 |
| --- | ---: | ---: |
| ACS | 1.000000 | 0.999653 |
| NLTCS | 1.000000 | 0.999281 |
| BR2000 | 1.000000 | **0.655962** |
| Adult | **0.845572** | **0.670850** |

Interpretation:

1. The exact full-query-L2 allocator is byte-equivalent up to floating error
   for ACS/NLTCS/BR2000, so it preserves the current binary success and does
   not claim to solve BR2000.
2. On Adult it can reduce the declared reconstructable-query variance risk by
   `15.44%` without private adaptation or truth-derived tuning.
3. Equal-group L2 exposes a larger `32.91%` Adult and `34.40%` BR2000 public
   risk opportunity, but it changes the estimand. It should not be promoted
   until its relationship to AvgTVD and the primary composite is proved.

For Adult full-query L2, the optimal pair-rho share moves from `91.98%` to
`84.66%`; several one-way anchors receive about `2.3x--3.0x` their current
rho. This follows from public mixed/prefix/range reuse, not from observed true
errors.

## 5. Released sparse-tree stability

Two predeclared scores were audited:

```text
total_excess:
  [||z_e||^2 / v_e - m_e]_+

null_standardized:
  [||z_e||^2 / v_e - m_e]_+ / sqrt(2 m_e)
```

Mean pairwise edge Jaccard over seeds:

| Dataset | epsilon | Total excess | Null standardized |
| --- | ---: | ---: | ---: |
| ACS | 0.1 | 0.321 | 0.321 |
| ACS | 0.3 | 0.543 | 0.543 |
| NLTCS | 0.1 | 0.437 | 0.437 |
| NLTCS | 0.3 | 0.615 | 0.615 |
| BR2000 | 0.1 | 0.629 | 0.371 |
| BR2000 | 0.3 | 0.816 | 0.738 |
| Adult | 0.1 | 0.867 | 0.335 |
| Adult | 0.3 | 1.000 | 0.586 |

The high total-excess stability on Adult/BR2000 is not enough to authorize a
tree prior. The two score definitions select very different edge geometries:

```text
Adult score-profile edge overlap by seed:   0.077--0.167
BR2000 score-profile edge overlap by seed:  0.083--0.130
```

Mean selected edge dimensions:

| Dataset/epsilon | Total excess | Null standardized |
| --- | ---: | ---: |
| Adult 0.1 | 359.3 | 172.4 |
| Adult 0.3 | 377.1 | 179.4 |
| BR2000 0.1 | 59.2 | 7.6 |
| BR2000 0.3 | 59.3 | 7.2 |

Total excess is stable partly because it strongly favors aggregate signal in
large-dimensional edges; null-standardized evidence removes much of that
effect and is less stable. Choosing one after seeing utility would be
post-hoc. A tree prior therefore still needs a theorem-level edge risk before
it can become the next experiment.

## 6. Decision

The local route ordering changes to:

```text
1. exact full-reconstructable-workload L2 allocation pilot on Adult
2. expert proof/decision on equal-group or TVD-related public risk
3. released sparse-tree prior only after a dimension-calibrated edge theorem
```

The first route is not a new heuristic. It completes the expert document's
original public allocation formula for the actual public workload. It also has
a clean causal test because it is identical to current allocation on
ACS/NLTCS/BR2000 and changes Adult for a fully public reason.

## 7. Recommended WP10a gate

Before reading any new utility, freeze:

```text
dataset:        Adult
epsilon:        0.1
seed:           0
arms:
  current cell-reference public_optimal
  full-reconstructable-query-L2 public_optimal
same:
  total rho
  standard-normal noise coupling
  QDTE initialization
  exact precision objective
  5000 iterations
  4096 candidates/iteration
```

Required mechanism checks:

```text
public workload hashes frozen
cell-reference importance identity passes
new allocation equals closed-form optimum
actual rho spent equals declared rho
no true utility available during generation
all QDTE invariants and batch certificates pass
```

Suggested development gate:

```text
primary geometric ratio <= 0.97
no primary metric > 1.05
MaxTVD and MaxError <= 1.15
```

Failure freezes the exact workload-L2 allocator. Success authorizes Adult
epsilon `0.3` seed0, then a predeclared multi-seed confirmation. It does not
authorize a sweep over query/group mixtures.
