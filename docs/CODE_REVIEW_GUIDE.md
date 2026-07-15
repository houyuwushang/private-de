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
qdte/measurement/public_artifact.py
scripts/measure_qdte_transcript.py
scripts/generate_qdte_from_transcript.py
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

The canonical external runner enables a fail-closed release profile: it loads
the public codebook/binning from `schema.json`, requires public `n_rows`, fixes
add/remove adjacency, disables in-process true evaluation, and reports privacy
from actual `rho_spent`. The resolved config stores the explicit value in
`privacy.public_n_rows`, verifies it against the input row count, and writes a
per-Gaussian-vector ledger containing sensitivity, noise scale, and charged
rho. See
`configs/variants/dp_release_profile_overlay.yaml` for the declarative form.

The deployable two-process entry points are
`scripts/measure_qdte_transcript.py` and
`scripts/generate_qdte_from_transcript.py`. The first process alone receives
the private CSV. The second requires a hash-sealed public transcript with a
validated actual-spend ledger and cannot accept `run.input_csv` or
`init.encoded_npy`. Byte-equivalence and fail-closed tamper tests live in
`tests/test_public_transcript_generation.py`.

The research runners also compute `full_true_*` evaluator metrics and may log
`private_*` selector diagnostics. Those files are internal offline diagnostics,
not releasable DP outputs. A deployable release must export only the declared
DP transcript/synthetic data and must omit these diagnostic fields.

See `docs/DP_BOUNDARY.md` for the complete boundary.

## Focused Verification

```bash
conda run -n qdte pytest -q \
  tests/test_edit_advantage.py \
  tests/test_transport.py \
  tests/test_preprocess.py \
  tests/test_config_validation.py \
  tests/test_public_transcript_generation.py \
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

python3 scripts/audit_reproducibility_docs.py
python3 scripts/verify_public_release.py
```

The experiment runners require encoded public-workload inputs that are not
committed to this repository. `docs/REPRODUCIBILITY.md` describes the input
contract.

## Frozen Research Status

The source surface intentionally includes experimental selector, stronger
projection, projection-aware uncertainty, interaction-measurement, and
structured-neighborhood paths so their implementation can be reviewed. Their
presence does not make them paper defaults:

- `QDTE-Standard` with the public static measurement schedule remains the
  paper-facing default;
- private adaptive selection remains diagnostic and is not silently enabled;
- nonnegative consistency is a supporting projection proposition/ablation,
  not a final-utility guarantee;
- projection-aware diagonal uncertainty is a non-default variant with an
  explicit tail tradeoff; and
- structured two-row search remains a controlled generator capability rather
  than the deployed default.

The clean public repository contains source, configs, focused tests, and these
concise review documents. Private experiment outputs, handoff history, expert
discussion, and post-hoc development notes remain outside that release
surface.
