# SAGE-QDTE-ICE WP10b GCEA Public-Gate Result

Date: 2026-07-15

## 0. Decision

The frozen public-only GCEA mechanism gate failed. No private measurement and
no synthetic generation are authorized.

```text
protocol: SAGE-QDTE-ICE-WP10B-GCEA-20260715-v1
candidate: SAGE-QDTE-Static-ICE-GCEA-v1
mechanism_gate_passed: false
generation_authorized: false
decision: q1_a_freeze_ice_binary_profile
```

Per the expert's explicit fallback, this result closes general-cardinality ICE
allocator research for the current paper:

```text
ICE:  binary / low-cardinality profile
QDTE: broad main method
```

The threshold will not be relaxed, the public gate will not be rerun, and no
sparse-tree, family-weight, P3, entropy, shrinkage, cycle, selector, or tail
rescue arm will be added.

## 1. Boundary

The gate read only the frozen public files:

```text
schema.json
metadata.json
queries_full.json
workload_groups.json
```

It did not load a private table, released measurement transcript, synthetic
table, true answer, or offline utility. It performed allocation and certificate
calculation only.

## 2. Public results

The reported `t*` is the stage-one solver result for the minimax of the public
MAE, RMSE, and AvgTVD risks under hard measurement-level MaxError and MaxTVD
envelopes.

| Dataset | epsilon | reported t* | MAE ratio | RMSE ratio | AvgTVD ratio | MaxError envelope | MaxTVD envelope | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Adult | 0.1 | 0.9940343920 | 0.9921067101 | 0.9940343921 | 0.9940343921 | 0.9770828903 | 0.9461604240 | no |
| Adult | 0.3 | 0.9940343920 | 0.9921067242 | 0.9940343921 | 0.9940343921 | 0.9770829714 | 0.9461606710 | no |
| BR2000 | 0.1 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | no |
| BR2000 | 0.3 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | no |

The required public primary threshold was `t* <= 0.97`. Adult's evaluator
envelope changes the allocation and improves all five public risks, but its
worst primary surrogate improves by only about `0.60%`, not the required `3%`.
BR2000 remains at the control allocation in the frozen solver run.

Therefore both datasets fail the decisive primary-risk threshold independently
of any end-to-end utility.

## 3. Other certificate checks

Public reconstruction was exact to numerical precision:

```text
Adult reconstruction max abs:  1.0547118733938987e-15
BR2000 reconstruction max abs: 6.661338147750939e-16
required:                      <= 1e-10
```

The tail envelope constraints passed in every cell. They did not cause a tail
regression; they instead removed most of the allocation freedom needed for a
large simultaneous primary improvement.

The full gate nevertheless had additional numerical-certificate failures:

```text
Adult epsilon 0.1 final KKT gap: 1.9303365024825325e-4
Adult epsilon 0.3 final KKT gap: 5.279664474073797e-4
BR2000 stage-one KKT multipliers: degenerate at the control/t=1 boundary
Adult cross-epsilon share max abs: 5.961670464932345e-8 > 1e-10
```

These failures cannot be used as a reason to rerun: the protocol defines any
certificate failure as Q1-A, and the independent `t* <= 0.97` gate already
fails for all four cells. Because the BR2000 KKT certificate is not valid, this
document does not claim a new theorem that `t*=1` is the exact mathematical
optimum; it claims only the predeclared mechanism gate failed and generation is
not authorized.

## 4. Public coverage

| Dataset | Evaluator queries | Reconstructable queries | Unsupported queries | Complete reconstructable TVD blocks | Strategy blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adult | 28,654 | 22,534 | 6,120 | 126 | 120 |
| BR2000 | 8,907 | 4,811 | 4,096 | 105 | 105 |

All strategy blocks were movable under the reconstructable public objective;
no hidden fixed-block rule determined the result.

## 5. Interpretation

GCEA answered the authorized question without spending privacy budget:

> Exact evaluator aggregation and measurement-level tail envelopes do not
> expose the predeclared 3% simultaneous public-risk improvement on both
> general-cardinality datasets under the frozen strategy and control.

WP10a showed that unconstrained full-query L2 reallocation had average gains
but unsafe pointwise tail behavior. WP10b shows that enforcing both evaluator
alignment and tail safety leaves too little certified gain to justify another
end-to-end panel. Under the frozen decision rule, the binary/general-cardinality
split is now the paper-facing method boundary.

This result does not change any existing QDTE objective, DP invariant, WP9
utility, or paper-facing baseline result.

## 6. Evidence

```text
protocol:
  docs/SAGE_QDTE_ICE_WP10B_GCEA_PROTOCOL_20260715.md

public gate:
  outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.json
  outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.csv

seal:
  outputs/static_ice_wp10b_gcea_public_gate_20260715/sealed_manifest.json
```

All three sealed artifacts and all thirteen frozen source/public-input records
were rehashed after the run with no mismatch.
