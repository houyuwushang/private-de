# QDTE Differential-Privacy Boundary

This document states the data-flow boundary enforced by the public QDTE code.
It distinguishes the released mechanism from offline evaluation.

## 1. Public Inputs

Before private data are accessed, a run fixes:

- the encoded schema and attribute cardinalities;
- query definitions and measurement groups;
- sensitivity bounds and the zCDP budget split;
- projection and clipping rules;
- the synthetic row count;
- QDTE candidate, transport, stopping, and random-seed configuration.

Cardinalities, category codebooks, numerical bin edges, missing-value rules,
and the row count are treated as public metadata. Research runs may infer a
schema for convenience, but such a run is not a deployable DP release profile.
With `privacy.dp_release_mode=true`, validation fails closed unless:

- `preprocess.public_schema_json` names an explicit public schema;
- `privacy.public_row_count=true` declares `n` public and
  `privacy.public_n_rows` supplies its positive integer value;
- `privacy.adjacency=add_remove` fixes the adjacency used by the sensitivity
  and accounting proofs; and
- all in-process true-data evaluation is disabled.

`configs/variants/dp_release_profile_overlay.yaml` records these requirements.
The canonical external runner additionally reads `n_rows` from public input
metadata rather than opening `real_encoded.npy` to configure the run. The
active runner verifies that the private input has exactly this declared row
count before measurement and uses the declared value for projection and the
default synthetic row count.

For deployment, the repository also exposes a process-separated path:

```text
private measurement process:
  private CSV + public schema/workload -> sealed public transcript

public generation process:
  sealed transcript + public generation config -> synthetic table
```

The sealed transcript contains only `schema.json`, `queries.json`,
`measurements.json`, and `transcript_manifest.json`. The manifest hashes the
three payloads and certifies the add/remove privacy ledger and public row
count. `run.transcript_only_generation=true` forbids both `run.input_csv` and
`init.encoded_npy`; the generation process verifies the manifest before QDTE
starts and never invokes the private-data loader.

## 2. Private Measurement

For each public measurement group `j`, the mechanism releases a Gaussian noisy
answer with variance calibrated to its public L2 sensitivity and allocated
zCDP budget:

```text
rho_j = sensitivity_l2_j^2 / (2 * noise_variance_j)
sum_j rho_j <= rho_total
```

The serialized ledger records both declared `rho_total` and actual
`rho_spent`; reported `epsilon_delta` is computed from `rho_spent`.
Each Gaussian-vector ledger entry records its scope label, add/remove
adjacency, L2 sensitivity, sigma multiplier, realized noise standard
deviation, and charged rho under `gaussian_zcdp_exact_v1`.

Projection, clipping, partition repair, variance post-processing, and row
generation are post-processing when they read only these releases and public
information.

### Optional Certified Private Selection

The default static profile has no private selection step. A transcript-only
adaptive selector is also post-processing and spends no selection privacy.

The experimental private-selection path may access exact private block answers
only inside a registered exponential mechanism. That path is accepted only
when all of the following are explicit:

- a fixed public candidate set;
- a proved round-conditional global sensitivity;
- exponential-mechanism sampling rather than an unnoised private argmax;
- a charged selection ledger composed with Gaussian measurement spending; and
- no exact private score entering QDTE or a released output artifact.

Conditional on the previous DP transcript, a base measure computed only from
that transcript does not change the current private-score sensitivity. The
research implementation records this distinction, but the conditional privacy
argument and any score used for a paper claim must still be reviewed together
with the exact adaptive-composition protocol.

## 3. QDTE Objective

The implementation preserves these invariants:

```text
residual[q] = target_projected[q] - answer_syn[q]
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
delta[q] = phi_q(x_new) - phi_q(x_old)
edit_advantage = delta @ (residual * inv_variance)
                 - 0.5 * ((delta * delta) @ inv_variance)
                 - lambda_cost * edit_cost
```

In `privacy.mode=dp`, candidates, exact delta scoring, transport, stopping, and
config selection may use only released noisy/projected measurements, their
released variances, public queries/schema, synthetic state, and internal
randomness.

They must not read exact private answers or offline utility metrics.

## 4. Standard and Structured

`QDTE-Standard` is the end-to-end DP default. `QDTE-Structured` adds exact
aggregate-delta two-row transport units and released-residual triggers. It uses
the same released objective and consumes no additional privacy budget.

Structured search can fit released noise more aggressively. This is a utility
and generalization issue, not an additional privacy query. The paper therefore
keeps Structured as a controlled-generator profile rather than silently
replacing Standard.

## 5. Gaussian Fission and Refit

`QDTE-FissionRefit` draws fresh auxiliary Gaussian randomness and derives train
and validation views from an already released Gaussian measurement. Checkpoint
selection reads the validation view, and the refit reads the original released
target. Neither stage reads exact private answers.

Fission is randomized post-processing of the original release. It does not
create a second private measurement or spend a second privacy budget.

## 6. Offline Evaluation

Only after a complete synthetic table exists may the evaluator compute exact
answers on `real_encoded.npy` and `synthetic_encoded.npy`. It reports pointwise
MAE/RMSE/MaxError and block AvgTVD/MaxTVD after normalizing each answer vector
by its table row count.

Outside a sensitivity-certified and privacy-charged selector, these exact true
answers are offline evaluation artifacts only. They must not feed back into:

- query or scope selection;
- candidate generation or scoring;
- transport or stopping;
- checkpoint/config/hyperparameter selection;
- release decisions based on private utility.

Research runs may persist `full_true_*` metrics and `private_*` selector
diagnostics for internal auditing. These fields are not DP releases and must be
removed from any public result artifact. The public source repository includes
the runners so that this boundary can be reviewed; publishing source code is
not equivalent to publishing those private diagnostic outputs.

## 7. Controlled No-Noise Diagnostics

The same-target no-noise QDTE/GSD comparison is a generator diagnostic, not an
end-to-end DP release. Its exact target is used only under that explicitly
non-DP experimental protocol. It does not weaken the DP boundary of the primary
QDTE-Standard runs.

## 8. Enforcement

The public checks include:

```bash
conda run -n qdte pytest -q tests/test_dp_boundary_no_true_answers_in_generator.py
conda run -n qdte pytest -q tests/test_public_transcript_generation.py
conda run -n qdte pytest -q tests/test_preprocess.py tests/test_config_validation.py
conda run -n qdte pytest -q tests/test_edit_advantage.py
conda run -n qdte pytest -q tests/test_transport.py
conda run -n qdte pytest -q tests/test_measurement_fission.py
conda run -n qdte pytest -q tests/test_run_integrated_sage_qdte.py
conda run -n qdte pytest -q tests/test_run_orthogonal_low_budget_pilot.py
conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2
```

The DP smoke logs the privacy ledger and prints a separate marker immediately
before exact true answers are opened for offline evaluation.
