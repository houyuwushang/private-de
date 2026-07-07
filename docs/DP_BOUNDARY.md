# Differential Privacy Boundary

This document states the DP boundary enforced by SAGE/QDTE. It is intended for public reproducibility and reviewer audit.

## 1. Core Rule

In:

```text
privacy.mode = dp
```

QDTE optimizes only against noisy or projected measurements and their public variances. Exact true answers are allowed only in the offline evaluator after a synthetic table has already been produced.

## 2. Generator Invariants

The row generator must preserve the following objective definitions:

```text
residual[q] = target_projected[q] - answer_syn[q]
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
delta[q] = phi_q(x_new) - phi_q(x_old)
edit advantage =
  delta @ (residual * inv_variance)
  - 0.5 * ((delta * delta) @ inv_variance)
  - lambda_cost * edit_cost
```

For paper-facing strong configs:

```text
qdte.objective_weighting = variance
```

so:

```text
inv_variance[q] = 1 / variance[q]
```

after measurement/projection uncertainty propagation.

## 3. Static Measurement Route

The strict main SAGE/QDTE table uses:

```text
privacy.measurement_mode = static_all
```

The measured workload schedule and budget split are public. There is no data-dependent query selection in this route.

The privacy cost is Gaussian zCDP measurement:

```text
rho_total = sum_j Delta_2(M_j)^2 / (2 * sigma_j^2)
```

Projection and QDTE generation are post-processing.

## 4. Projection Boundary

Measurement post-processing may include:

```text
partition simplex projection
non-partition clipping
prefix monotonicity projection
optional consistency projection
projection-aware variance propagation
```

These operations are privacy-free only because they consume noisy measurements, public query definitions, public constraints, and public variances. They must not access exact true answers in DP mode.

## 5. Disallowed True-Answer Uses

In DP mode, exact true answers must not be used for:

```text
active query selection
candidate generation
candidate scoring
transport
stopping
hyperparameter selection
generation-time logging that affects control flow
```

They also must not be written into `measurements.json`.

## 6. Allowed True-Answer Uses

Exact true answers may be used for:

```text
Gaussian measurement of selected public blocks
offline external evaluation
offline diagnostic reports after a run is complete
```

The external evaluator may read:

```text
true_answers_cache.npz
```

but the SAGE/QDTE run itself must not use this cache for active decisions.

## 7. SAGE-Select Privacy Routes

SAGE-Select has two distinct privacy routes.

### Exponential-Mechanism Route

If the proof-closed ordered-gain SAGE-Select score is evaluated directly on private data, the unit-sensitivity theorem permits exponential-mechanism selection:

```text
Pr[M] proportional to mu(M) * exp(epsilon_t * S(D, M) / 2)
```

Conservative zCDP accounting:

```text
rho_select,t <= epsilon_t^2 / 2
```

Sharper bounded-range EM accounting, if invoked:

```text
rho_select,t <= epsilon_t^2 / 8
```

The selected block still pays Gaussian measurement cost:

```text
rho_measure,t = Delta_2(M_t)^2 / (2 * sigma_t^2)
```

### Transcript-Only Route

The certified adaptive runner uses transcript-only scoring:

```text
selection_input = transcript
selection_ledger = measurement_only
selection_rule = argmax
```

The score is computed from previous noisy/projected transcript answers, public variances, public query definitions, and current synthetic answers. It does not use exact true answers for candidate scoring.

Selection is therefore post-processing:

```text
rho_select,t = 0
```

Only the selected measurement pays privacy.

These privacy routes certify the proof-closed ordered-gain SAGE-Select score.
They do not certify exploratory rank-mask or other VOI scores unless a separate
sensitivity proof is supplied.

## 8. External Baseline Boundary

External baselines may use their own DP mechanisms. For fair comparison, they must output a row-level:

```text
synthetic_encoded.npy
```

Then the shared offline evaluator computes true utility metrics. Those true metrics must not feed back into the baseline run or SAGE run.

Upstream original-protocol reproduced baselines, such as RAP++ official and
PrivMRF official, may use native metrics and data interfaces. They must be
reported separately from the strict same-protocol row-level table.

## 9. Audit Checks

Before treating a run as DP-valid, check:

```text
privacy.mode = dp
privacy.measurement_mode is public or certified
evaluation.compute_true_query_error is false inside active SAGE external runs
synthetic generation reads only measurements/projections/variances
offline evaluation happens after synthetic_encoded.npy exists
```

For public release, the core invariant tests are in:

```text
tests/test_edit_advantage.py
tests/test_measurement.py
tests/test_external_evaluator.py
tests/test_run_sage_external.py
tests/test_engine_smoke.py
```
