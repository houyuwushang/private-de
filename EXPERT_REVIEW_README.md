# SAGE-QDTE RCE C1 Expert Review Snapshot

Start with:

- [C1 result and decision questions](docs/SAGE_QDTE_RCE_C1_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md)
- [Frozen C0/C1 protocol](docs/SAGE_QDTE_RCE_C0_C1_PROTOCOL_20260715.md)
- [Expert RCE method-limit route](docs/SAGE_QDTE_RCE_方法极限路线_20260715.md)
- [Frozen offline evaluation](outputs/sage_qdte_rce_c1_v3_eval_20260715/summary.json)
- [Post-seal confidence/KL diagnosis](outputs/sage_qdte_rce_c1_v3_postseal_20260715/summary.json)
- [Sealed blind-panel manifest](outputs/sage_qdte_rce_c1_v3_20260715/sealed_panel_manifest.json)

The twelve-cell same-transcript panel was fully generated and sealed before
one frozen offline evaluation. RCE-v1 did not improve WP9 and did not beat
Official AIM. The post-seal diagnostic found truth inside all twelve confidence
sets, while RCE moved much closer to the released one-way product prior and
discarded useful interaction structure. The decision packet asks whether to
stop Adult rescue or authorize exactly one richer released prior, dual-driven
measurement refinement, or fully specified ceiling experiment.

This snapshot includes implementation, focused tests, sanitized manifests,
per-cell runtime/certificates, and aggregate true-utility metrics. It excludes
private rows, raw CSV, exact-answer caches, raw/released measurement payloads,
query payloads, initial tables, and synthetic tables. Original source hashes
are retained in `EXPERT_REVIEW_SOURCE_HASHES.json`.
