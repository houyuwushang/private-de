# QDTE Paper Claim Matrix

Date: 2026-07-11

This document freezes the paper-facing terminology, contribution hierarchy,
allowed claims, evidence boundaries, and forbidden extrapolations before the
latest result package and manuscript are generated.

The machine-readable companion is
`docs/QDTE_PAPER_CLAIM_MATRIX_20260711.json`.

## Terminology freeze

| Paper-facing name | Development / artifact aliases | Paper role |
| --- | --- | --- |
| `QDTE-Standard` | `SAGE`, `QDTE-Base`, static SAGE/QDTE | Default end-to-end DP method |
| `QDTE-Structured` | Structured-v2, Structured-SA, structured atomic swap | Strong controlled generator profile |
| `QDTE-FissionRefit` | Structured-FissionRefit-v2, validation-minimum refit | Qualified MAE/RMSE alignment variant |
| `QDTE-RTP-local` | single-scope RTP/local feasible target | Transfer diagnostic only |
| `Query-LSQ` | `query_space_lsq` | Projection diagnostic only |
| `Teacher target` | realizable teacher-target test | Optimizer/transfer diagnostic only |

`Base` may appear in development notes and pairwise diagnostic labels, but the
manuscript uses `QDTE-Standard`. `SAGE` is not a paper-facing method name after
the rewrite. Heterogeneous queries remain the workload setting.

## Contribution hierarchy

### Contribution 1: Exact residual-directed row evolution

QDTE converts released noisy/projected query residuals into the exact
finite-difference decrease for variance-aware row edits. This is the primary
algorithmic contribution.

### Contribution 2: Structured atomic row transport

QDTE-Structured expands the directed neighborhood from ordinary row edits to
exact aggregate-delta atomic two-row transport. It is the strongest controlled
generator profile and remains part of the same QDTE family.

### Contribution 3: Noisy-target alignment and transfer diagnosis

FissionRefit provides a qualified MAE/RMSE alignment variant. Projection,
teacher-target, RTP, and T/F/C/E diagnostics expose why stronger released
target fit does not automatically become every final utility metric.

The heterogeneous workload catalogue is the experimental interface, not a
separate primary method contribution. SAGE-Select is not part of the main
contribution stack for this paper.

## Claim matrix

### QDTE-C1: End-to-end DP utility

Role: `main`.

Allowed claim:

> Under the shared four-dataset, seed0--4, rho=1 row-level protocol,
> QDTE-Standard has lower error than AIM, MST, and RAP softmax on all 20
> dataset-metric cells and lower error than high-power Private-GSD on 16 of 20
> cells. The four GSD wins are ACS/BR2000 AvgTVD and MaxTVD.

Required evidence:

```text
paper_package_seed0to4_20260706
primary run evidence audit: 100 completed runs
shared query/schema hashes and external evaluator
```

Forbidden extrapolation:

```text
QDTE dominates every GSD configuration or every utility metric.
The static main table is an adaptive-selection result.
```

### QDTE-C2: Exact edit advantage

Role: `main-theory`.

Allowed claim:

> For residual `r = target_projected - answer_syn` and answer change `delta`,
> QDTE's accepted edit score is the exact finite-difference decrease of the
> weighted quadratic released-measurement objective, minus public edit cost.

Required evidence:

```text
algebraic identity in Method/Analysis
objective invariant tests
incremental answer drift checks
```

Forbidden extrapolation:

```text
The edit is guaranteed to reduce exact private-truth error.
```

### QDTE-C3: Privacy boundary

Role: `main-theory`.

Allowed claim:

> With public static measurement groups and a valid Gaussian zCDP ledger,
> projection and QDTE generation are post-processing because they use only
> released measurements, public queries/variances, synthetic state, and
> internal randomness.

Required evidence:

```text
static privacy theorem
DP-boundary tests
config and measurement audits
```

Forbidden extrapolation:

```text
Offline exact true answers may influence candidates, scoring, transport,
stopping, or hyperparameter selection.
```

### QDTE-C4: Controlled generator capability

Role: `main-mechanism`.

Allowed claim:

> On the same exact mixed no-noise target for the three tested seed0 datasets
> (ACS, BR2000, and Adult), QDTE-Structured reaches lower target loss than the
> fully seeded official GSD implementation and wins all 15 corresponding
> offline utility cells.

Required evidence:

```text
qdte_structured_optimized_gate1_v1
qdte_gsd_converged_no_noise_v1
qdte_gsd_no_noise_breadth_v1
official GSD full seeding audit
endpoint and time-to-quality reconstruction
```

Mandatory caveat:

> Adult is both better and faster in the observed run. ACS and BR2000 require
> more QDTE wall time to first pass GSD's final endpoint, so endpoint strength
> is not a universal speed claim.

Forbidden extrapolation:

```text
The result holds over multiple seeds.
The oracle/no-noise experiment is an end-to-end DP result.
QDTE-Structured is always faster than GSD.
```

### QDTE-C5: Structured noisy-target boundary

Role: `main-diagnostic`.

Allowed claim:

> Structured QDTE lowers released measured loss on all four fixed-noise seed0
> datasets, but fails the predeclared end-to-end DP utility replacement gate.
> Search-width calibration improves alignment but still does not make
> Structured the DP default.

Required evidence:

```text
qdte_structured_standard_v2_optimized_fixed_noise_seed0
qdte_structured_search_aware_v1_fixed_noise_seed0
predeclared promotion-gate documents
```

Forbidden extrapolation:

```text
More generator search necessarily improves true utility.
Structured replaces QDTE-Standard in the DP main table.
```

### QDTE-C6: FissionRefit L2 alignment

Role: `qualified-variant`.

Allowed claim:

> In the predeclared seed3 four-dataset confirmation, QDTE-FissionRefit wins
> MAE on 4/4 datasets and RMSE on 3/4, for 7/8 combined wins; the largest L2
> regression is BR2000 RMSE at 0.62%, below the frozen 2% gate.

Required evidence:

```text
qdte_structured_sa_terminal_fixed_noise_seed3
qdte_structured_fission_refit_v2_fixed_noise_seed3
completed controls and pipelines before true evaluation
```

Mandatory caveat:

> FissionRefit is a two-pass, non-default MAE/RMSE variant. It does not
> consistently improve AvgTVD, MaxTVD, or MaxError.

Forbidden extrapolation:

```text
FissionRefit is universally utility-improving.
Development seed1 is confirmatory evidence.
The refit path inherits an exact checkpoint-selection theorem.
```

### QDTE-C7: Projection-to-generation transfer gap

Role: `diagnostic/future-work`.

Allowed claim:

> Better query-space or local row-realizable targets can improve target and fit
> loss without uniformly improving final row-level utility. Teacher-target
> evidence shows QDTE can fit a fully row-realizable target much more tightly,
> making global target interaction and metric mismatch more plausible primary
> bottlenecks than a universally weak optimizer.

Required evidence:

```text
query-space LSQ target/fit/final decomposition
RTP-local fixed diagnostics
teacher-target test
T/F/C/E decomposition
```

Forbidden extrapolation:

```text
RTP-local solves the transfer gap.
Better projected target quality guarantees final utility.
The diagnostic identifies a unique causal mechanism.
```

### QDTE-C8: Row-answer-polytope interpretation

Role: `main-theory`.

Allowed claim:

> QDTE can be viewed as an integer pairwise row-edit analogue of column
> generation or conditional gradient over the row-answer polytope. Its score is
> an exact finite difference on the discrete row multiset.

Required evidence:

```text
row-answer-polytope formulation
single-edit and aggregate-delta identities
appropriate conditional-gradient/column-generation citations
```

Forbidden extrapolation:

```text
QDTE is exactly standard continuous Frank-Wolfe.
QDTE inherits standard Frank-Wolfe convergence guarantees.
```

### QDTE-C9: Joint utility remains unresolved

Role: `appendix/limitation`.

Allowed claim:

> Exact answer-space mixtures show that five-metric Pareto improvements can
> exist, but tested fixed-N rounding, TVD-L1, and released-validation selection
> do not reliably convert them into a single fixed-row method.

Required evidence:

```text
QDTE_JOINT_UTILITY_RESULTS_20260711.md
pool-rounding artifacts
correct Structured TVD-L1 artifacts
released-validation mixture reports
```

Forbidden extrapolation:

```text
The paper solves all five utility metrics jointly.
Expanded-row mixtures are a fair default without a changed protocol.
```

## Placement freeze

Main body:

```text
QDTE-C1, C2, C3, C4, C5, C6, C8
compact C7 diagnosis
```

Appendix/limitations:

```text
detailed C7 evidence
C9 negative evidence
baseline provenance and full tables
```

Not in the main contribution stack:

```text
SAGE-Select
PA-Diag16 as a default
RTP-local as a method
TVD-L1 as a utility solution
```
