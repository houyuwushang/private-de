# Project Rules

- Preserve the QDTE objective invariants:
  - `residual[q] = target_projected[q] - answer_syn[q]`
  - `measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]`
  - `delta[q] = phi_q(x_new) - phi_q(x_old)`
  - `edit advantage = delta @ (residual * inv_variance) - 0.5 * ((delta * delta) @ inv_variance) - lambda_cost * edit_cost`
- Preserve the DP boundary. In `privacy.mode=dp`, QDTE optimizes only against noisy/projected measurements and their variances.
- In `privacy.mode=dp`, exact true answers may be computed only for offline evaluation metrics. They must not be used for active query selection, candidate generation, scoring, transport, stopping, or hyperparameter selection.
- Every task must update `docs/HANDOFF.md` with changed files, tests run, current status, and the next recommended task.
