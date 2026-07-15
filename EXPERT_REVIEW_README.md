# SAGE-QDTE-ICE WP9/WP10a Expert Review Snapshot

This branch is a self-contained, sanitized review snapshot. Start with:

- [WP10a result and the A/B decision request](docs/SAGE_QDTE_ICE_WP10A_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md)
- [WP9 four-dataset confirmation](docs/SAGE_QDTE_ICE_WP9_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md)
- [WP9a public-only diagnostic](docs/SAGE_QDTE_ICE_WP9A_PUBLIC_ROUTE_DIAGNOSTIC_RESULT_20260715.md)
- [Frozen WP9 figure](outputs/static_ice_wp9_figures_20260715/wp9_static_ice_vs_aim_gate.png)

The frozen WP9 result is regime-specific: Static-ICE beat Official AIM on
NLTCS and ACS at epsilon 0.1/0.3, and lost on BR2000 and Adult. It won 4/8
dataset-epsilon cells. WP10a reduced its exact declared public L2 measurement
risk by 15.44% on Adult, but failed the frozen final-utility and tail gates.

The snapshot includes aggregate/offline evaluation summaries, plots, sealed
manifests, the relevant implementation, focused tests, and the four public
schema/metadata pairs needed by the panel-plan test. It intentionally omits
private rows, exact-answer caches, raw measurement/query payloads, raw CSVs,
and large synthetic tables. Sealed manifests retain the original SHA-256
records for omitted artifacts. `EXPERT_REVIEW_SOURCE_HASHES.json` records the
hashes of the original local evidence files before path sanitization.

All paths in copied documents and JSON are repository-relative or explicitly
marked as omitted external inputs. This review branch is independent of the
paper-facing default branch and does not promote WP10a as a method variant.
