# RCE C1 Released-Only Restricted-Mixture Protocol

Date: 2026-07-15

```text
protocol_id: SAGE-QDTE-RCE-C1-RESTRICTED-MIXTURE-20260715-v2
role: post-seal estimator-vs-integer diagnostic
global_relaxed_certificate: false
```

The unsealed `v1` execution was stopped before offline evaluation because
SLSQP returned a boundary line-search status for a stage-one problem whose
optimum was already analytically certified: a declared component had zero
confidence slack, and slack is constrained to be nonnegative. Version `v2`
records this feasible-component certificate directly. No objective, support,
component, confidence threshold, or evaluation rule changed.

The first `v2` execution directory was also left unsealed before evaluation
when JAX attempted to use an occupied GPU while recomputing component query
answers. The formal execution pins this diagnostic to CPU; GPU hardware is not
part of the estimator and this changes no numerical definition.

The first frozen offline evaluator (`eval-v1`) stopped at its query-order gate
and emitted no metrics. It incorrectly required the measured confidence
catalogue to equal the larger full evaluator catalogue. `eval-v2` verifies the
stored fractional answers on the measured catalogue, then evaluates the same
frozen component weights by answering each released component table on the
full evaluator workload and taking their convex combination. No missing query
answer is inferred from the measured subset.

## Question

The formal C1 panel showed that integer RCE-v1 is worse than the frozen WP9
point-target control. This diagnostic asks a narrower question without adding
measurements, changing the confidence set, or reading private truth:

> On the convex hull of three already released-only empirical tables, does the
> exact minimum-product-KL RCE objective select the WP9 direction, or does it
> continue to prefer a more conservative and lower-utility distribution?

The result separates interpolation within a fixed released-only support from
the candidate-limited integer row-edit trajectory. It does not estimate the
global relaxed RCE optimum over the full row domain.

## Frozen cells

```text
datasets: Adult, BR2000
epsilon: 0.1, 0.3
seeds: 0, 1, 2
source: sealed SAGE-QDTE-RCE-C1-20260715-v3 panel
```

For each cell, the declared component distributions are exactly:

```text
p_initial      shared released-only initial table
p_wp9          frozen Static-ICE point-target QDTE output
p_rce          sealed RCE-v1 output
```

No component may be added or removed after evaluation. AIM output, real rows,
true answers, offline metrics, and evaluator feedback are unavailable to the
solver process.

## Restricted optimization problem

Let

```text
p(w) = w_initial p_initial + w_wp9 p_wp9 + w_rce p_rce
w >= 0
sum(w) = 1
```

and retain the exact C1 confidence operator and released one-way product prior.
The solver uses the same lexicographic objective as RCE-v1:

1. minimize the nonnegative maximum confidence violation `s_star`;
2. on the optimal-slack face, minimize `D_KL(p(w) || p0)`.

The KL is evaluated on the union of full-row atoms present in the three
empirical tables. It is not replaced by KL over three component labels.

The result must report:

```text
component weights
union-support size and hash
confidence slack and ratios
KL to released product prior
stage-one and stage-two solver status
fractional query-answer hash
global_certified = false
```

## Process and evaluation boundary

The released-only solve runs and seals all twelve cells before any fractional
answer is compared with truth. A separate offline evaluator may then read real
rows and report MAE, RMSE, AvgTVD, MaxTVD, and MaxError.

The diagnostic has no promotion authority and cannot select a new method,
prior, confidence radius, support, checkpoint, or hyperparameter.

## Frozen interpretation

```text
restricted optimum moves materially toward WP9 and improves utility:
  evidence of an integer/search-path gap within the declared hull.

restricted optimum stays near RCE/initial and remains worse than WP9:
  direct evidence that minimum KL to the one-way product prior, rather than
  integer interpolation alone, rejects useful WP9 interaction structure.

any result:
  not a global relaxed ceiling and not authorization for C2 or C3.
```
