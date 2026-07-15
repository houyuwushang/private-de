# SAGE-QDTE-ICE WP10a Workload-Allocation Protocol

Date: 2026-07-15

Protocol ID:

```text
SAGE-QDTE-ICE-WP10A-WORKLOAD-ALLOCATION-20260715-v1
```

Candidate method ID:

```text
SAGE-QDTE-Static-ICE-WorkloadL2-v1
```

## 1. Question

Does completing the expert-defined public allocation

\[
c_a=\sum_{q\in Q_{\le2}}\|L_{qa}\|_2^2,
\qquad
\rho_a\propto s_a\sqrt{c_a}
\]

improve Adult low-budget final utility relative to the frozen WP9
cell-reference allocator?

This is an allocation-only causal pilot. It does not change the measured
orthogonal components, total privacy budget, target reconstruction, precision
operator, QDTE search profile, initialization, or evaluator.

## 2. Frozen Cell

```text
dataset: Adult
epsilon: 0.1
delta:   1e-9
seed:    0
adjacency: unbounded / add_remove
```

Frozen control:

```text
outputs/static_ice_wp9_formal_20260715/adult/epsilon_0.1/seed_0
```

The control is reused rather than rerun. Its mechanism gate already passed
5000 iterations and 20.48 million candidate scores without true utility.

## 3. Public Workload-L2 Allocation

Inputs:

```text
public schema.json
public queries_full.json
public workload_groups.json
```

Every full-workload query with public attribute scope order at most two is
factorized exactly through the one-way/pair orthogonal strategy. Every query
has unit weight. Unsupported high-order queries are recorded and make no
contribution; they are not approximated.

Frozen public expectations from WP9a:

```text
full queries:                   28,654
supported order <=2 queries:   22,534
unsupported order 3 queries:    6,120
cell-reference identity error: <= 1e-10
optimal/current risk ratio:     0.8455718531684706
current pair-rho share:         0.9197821438450282
workload pair-rho share:        0.8465777443293414
```

No family weight, group normalization, mixture coefficient, cardinality
exponent, or true-error feedback is allowed.

## 4. Coupled Mechanism

Both hypothetical mechanisms spend the full declared rho separately. They are
experimental alternatives, not a jointly released composition.

The candidate uses:

```text
same private measurement rows
same strategy block ordering
same measurement RNG seed as WP9 control
same standard-normal draw per coefficient
candidate-specific public coefficient variance
```

The runner reconstructs the WP9 control transcript in memory and verifies it
against the sealed control. For every block it also verifies:

\[
\frac{z_a^{\mathrm{control}}-\theta_a}{\sqrt{v_a^{\mathrm{control}}}}
=
\frac{z_a^{\mathrm{candidate}}-\theta_a}{\sqrt{v_a^{\mathrm{candidate}}}}
\]

to tolerance `1e-12`. Exact private coefficients are used transiently inside
the Gaussian measurement mechanism only and are never written to an artifact.

## 5. Frozen Generation

```text
initial table:
  frozen WP9 control synthetic_initial_encoded.npy

precision:
  OrthogonalInteractionPrecision

QDTE:
  5000 iterations
  4096 candidates per iteration
  20.48 million candidates total
  atom_flow batch transport
  no confidence stop
  no P3
  no shrinkage
  no entropy
  no cycles
  no fission
```

The generation seed is the frozen WP9 seed. The generated initial table must
hash-match the frozen control initial table.

## 6. No-Truth Seal

Before evaluation, the candidate must pass:

```text
public input and source hashes frozen before generation
control artifacts unchanged
cell-reference identity
workload factorization counts
closed-form allocation and risk certificate
rho spent equals rho declared
control transcript reproduction
standard-normal noise coupling
all component variances finite and positive
exact precision operator active
5000 iterations completed
20.48 million candidates scored
incremental answer drift <= 1e-8
initial table hash equality
no true/offline metric emitted
```

Only after the seal records `offline_evaluation_authorized=true` may the frozen
evaluator load exact data.

## 7. Frozen Utility Gate

Candidate/control ratios use the canonical full heterogeneous evaluator.

Primary composite:

```text
geometric mean of:
  full_true_mae
  full_true_rmse
  full_true_avg_tvd
```

Pass only if all hold:

```text
primary composite <= 0.97
each primary metric <= 1.05
full_true_max_tvd <= 1.15
full_true_max_error <= 1.15
```

Decision:

```text
pass:
  authorize Adult epsilon 0.3 seed0, then a separately frozen multiseed gate

fail:
  freeze exact full-workload-L2 allocation as a negative diagnostic
```

The result cannot authorize equal-group allocation, a TVD profile, a released
tree prior, or a broad default replacement.
