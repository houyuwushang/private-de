# SAGE-QDTE-ICE WP9a Public-Route Diagnostic Protocol

Date: 2026-07-15

Protocol ID:

```text
SAGE-QDTE-ICE-WP9A-PUBLIC-ROUTE-DIAGNOSTIC-20260715-v1
```

## 1. Purpose

WP9 exposed a stable cardinality regime: Static-ICE beat Official AIM on
NLTCS/ACS and lost on BR2000/Adult. WP9a does not run another utility
experiment. It answers two public/released-only questions before selecting a
new method:

1. Does the current allocation optimize the actual public evaluator workload,
   or only the complete one-/two-way cell sub-workload?
2. Are sparse trees selected from the released interaction transcript stable
   enough across DP seeds to be a credible structural prior?

No exact data, true query answer, synthetic utility, AIM output, or offline
evaluation cache may be loaded.

## 2. Frozen Inputs

```text
sealed WP9 panel:
  outputs/static_ice_wp9_formal_20260715/sealed_panel_manifest.json

public workload inputs:
  external_inputs/
    {nltcs,acs,br2000,adult}_sage_strong/
      schema.json
      queries_full.json
      workload_groups.json
```

Every measurement artifact must match its sealed path, size, and SHA-256.
Public workload file hashes are recorded in the output manifest.

## 3. Exact Public Workload Factorization

For a query whose public attribute scope has order at most two, enumerate its
Boolean predicate vector or matrix `f` over the public domain. Queries with
ordinary comparisons and linear halfspaces are both supported if their union
of attributes has order at most two.

For a one-way query on attribute `i`,

\[
q = \frac{n}{d_i}\mathbf 1^\top f
    + (C_i^\top f)^\top\theta_i.
\]

For a pair query on `(i,j)`, use the existing exact reconstruction maps
`L_i`, `L_j`, and `L_ij`:

\[
q = \frac{n}{d_i d_j}\mathbf 1^\top f
    + (L_i^\top f)^\top\theta_i
    + (L_j^\top f)^\top\theta_j
    + (L_{ij}^\top f)^\top\theta_{ij}.
\]

For public query weights `w_q`, each strategy block receives

\[
c_a = \sum_q w_q\|L_{qa}\|_2^2.
\]

The allocation risk and its closed-form optimum are

\[
R(c,\rho)=\sum_a\frac{c_as_a^2}{2\rho_a},
\qquad
\rho_a^* = \rho
\frac{s_a\sqrt{c_a}}{\sum_b s_b\sqrt{c_b}}.
\]

## 4. Frozen Public Risk Profiles

Three profiles are reported. They are diagnostics, not candidate methods.

### 4.1 `cell_reference`

```text
queries: families oneway and twoway only
weight:  1 per query
```

Its computed `c_a` must reproduce the existing
`StrategyBlock.public_importance` to numerical tolerance. Failure closes the
audit as an implementation error.

### 4.2 `full_reconstructable_query_l2`

```text
queries: every full-workload query with public scope order <= 2
weight:  1 per query
```

This is the exact public total-L2 risk for all queries reconstructable from the
current one-/two-way ICE strategy.

### 4.3 `equal_public_group_l2`

```text
queries: every full-workload query with public scope order <= 2
weight:  1 / total number of queries in its declared public group
```

Each public group has unit total weight when fully reconstructable. A partially
reconstructable group contributes only its supported public fraction; no
unsupported high-order query is silently approximated.

For profiles 4.2 and 4.3, report:

```text
supported query count and weight
unsupported count by family and scope order
R(profile, current WP9 rho)
R(profile, profile-optimal rho)
optimal/current risk ratio
rho movement by one-way versus pair blocks
largest public block allocation changes
```

No utility threshold is attached. A large public risk gap only authorizes a
separately predeclared allocation pilot.

## 5. Released Sparse-Tree Stability

For pair block `e` with released coefficient vector `z_e`, scalar coefficient
variance `v_e`, and dimension `m_e`, define

\[
X_e=\frac{\|z_e\|_2^2}{v_e},
\qquad
E_e=[X_e-m_e]_+.
\]

Two frozen released-only edge scores are audited:

```text
total_excess:       E_e
null_standardized:  E_e / sqrt(2 m_e)
```

For each dataset, epsilon, seed, and score, construct a deterministic maximum
spanning tree using lexicographic edge tie-breaking. Across seeds `0,1,2`,
report:

```text
pairwise edge Jaccard
three-seed intersection and union
edges selected in 3/3 and at least 2/3 trees
minimum selected-edge replacement margin
score-profile tree overlap
mean selected edge dimension
```

These scores are not declared optimal priors. They test whether released
pair-interaction evidence has enough repeatable structure to justify a later
tree-prior protocol.

## 6. Decision Use

WP9a may support one of three recommendations:

```text
allocation route:
  the full-workload optimal/current risk ratio shows substantial recoverable
  public risk, especially on Adult/BR2000.

tree-prior route:
  released trees have meaningful cross-seed stability at low epsilon.

regime-specific route:
  neither public risk recovery nor released tree stability is convincing.
```

WP9a itself cannot promote a method, alter QDTE Base, or justify any claim
against AIM.
