# SAGE-QDTE Code Review Guide

This branch is a code-focused research snapshot for external review. It does
not include private datasets, generated experiment outputs, manuscript drafts,
internal handoff notes, or expert discussions.

## Scientific Scope

The main mechanism is QDTE directed evolution. Given only released
noisy/projected query targets and their uncertainty, QDTE converts residuals
into exact finite-difference row-edit advantages:

```text
residual[q] = target_projected[q] - answer_syn[q]
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
delta[q] = phi_q(x_new) - phi_q(x_old)
edit_advantage = delta @ (residual * inv_variance)
                 - 0.5 * ((delta * delta) @ inv_variance)
                 - lambda_cost * edit_cost
```

SAGE-QDTE is the complete select/measure/project/generate pipeline around that
generator. The paper-facing default remains the static measurement profile and
`QDTE-Standard`. The adaptive private selector, nonnegative projection,
projection-aware uncertainty, and structured two-row search are reviewable
experimental paths; this snapshot does not silently promote them to defaults.

## Review Entry Points

Core QDTE:

```text
qdte/evolution/scoring.py       exact edit advantage
qdte/evolution/candidates.py    residual-directed proposals
qdte/evolution/transport.py     nonconflicting transport
qdte/evolution/engine.py        generator loop and invariants
```

Measurement and projection:

```text
qdte/measurement/measure.py
qdte/measurement/consistency.py
scripts/run_nonnegative_projection_pilot.py
scripts/reproject_measurements.py
```

Adaptive selection and the low-budget diagnostic harness:

```text
scripts/run_adaptive_selection_ablation.py
scripts/run_integrated_sage_qdte.py
scripts/audit_integrated_sage_qdte_run.py
scripts/run_orthogonal_low_budget_pilot.py
```

The low-budget runner supports complete one-way/two-way partition blocks,
explicit selection and Gaussian-measurement privacy accounting, and a
development-only conditional base measure in which a released-transcript SAGE
rank prior biases a sensitivity-one private partition-L1 exponential
mechanism.

## Privacy Boundary

QDTE never receives exact private answers. In DP mode, exact answers may be
accessed only by:

1. a Gaussian measurement mechanism with an explicit zCDP ledger; or
2. an explicitly enabled, sensitivity-certified exponential-mechanism
   selector with a charged selection ledger.

All candidate construction, edit scoring, transport, stopping, and
hyperparameter decisions use only released targets, released uncertainty,
public schema/workload metadata, and synthetic state.

The research runners also compute `full_true_*` evaluator metrics and may log
`private_*` selector diagnostics. Those files are internal offline diagnostics,
not releasable DP outputs. A deployable release must export only the declared
DP transcript/synthetic data and must omit these diagnostic fields.

See `docs/DP_BOUNDARY.md` for the complete boundary.

## Focused Verification

```bash
conda run -n qdte pytest -q \
  tests/test_edit_advantage.py \
  tests/test_dp_boundary_no_true_answers_in_generator.py \
  tests/test_run_integrated_sage_qdte.py \
  tests/test_run_orthogonal_low_budget_pilot.py \
  tests/test_consistency_projection.py \
  tests/test_nonnegative_projection_theorem.py \
  tests/test_reproject_measurements.py
```

Minimal DP smoke:

```bash
conda run -n qdte python scripts/smoke_qdte.py \
  --mode dp --rows 120 --max-iters 2
```

The experiment runners require encoded public-workload inputs that are not
committed to this repository. `docs/REPRODUCIBILITY.md` describes the input
contract.

## Current Review Questions

The accompanying private expert brief asks for decisions on:

- whether the adaptive selector should remain transcript-only or use a
  certified private exponential mechanism;
- how to derive a generation-aware private score with useful
  score-gap/sensitivity at very low privacy budgets;
- whether complete partition measurements plus certified nonnegative
  projection and projection-aware uncertainty should define a new low-budget
  variant;
- how to formulate and certify projection when unmeasured coordinates have
  exactly zero precision; and
- whether structured two-row search belongs in the current method or remains
  a fixed-target generator diagnostic.

This repository intentionally contains the implementation needed to inspect
those questions, but not the private experimental artifacts used to evaluate
them.
