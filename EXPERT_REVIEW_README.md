# SAGE-QDTE-ICE WP10b GCEA Expert Review Snapshot

Start with:

- [Adult post-GCEA decision request](docs/SAGE_QDTE_ADULT_POST_GCEA_EXPERT_DECISION_20260715.md)
- [WP10b public-gate result](docs/SAGE_QDTE_ICE_WP10B_GCEA_PUBLIC_GATE_RESULT_20260715.md)
- [Frozen WP10b protocol](docs/SAGE_QDTE_ICE_WP10B_GCEA_PROTOCOL_20260715.md)
- [Machine-readable public gate](outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.json)
- [WP9 four-dataset context](docs/SAGE_QDTE_ICE_WP9_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md)

The public-only GCEA gate failed before private measurement or synthetic
generation. Adult's reported minimax primary ratio was about 0.994, and
BR2000 returned the control allocation at 1.0. Both fail the predeclared
`t_star <= 0.97` threshold. Therefore `generation_authorized=false` and the
expert-specified fallback is Q1-A: ICE is frozen as a binary/low-cardinality
profile, while QDTE remains the broad main method.

This snapshot contains public schemas/metadata, aggregate evidence, sanitized
seals, implementation, and focused tests. It contains no private rows, raw
CSV, exact-answer cache, released measurement transcript, or synthetic table.
Original pre-sanitization hashes are retained in
`EXPERT_REVIEW_SOURCE_HASHES.json`.
