#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

try:
    from package_ice_wp10a_expert_review import (
        IDENTIFYING_TOKENS,
        LOCAL_TOKENS,
        ROOT,
        _iter_payload_files,
        file_record,
        read_json,
        sanitize_text,
        sanitize_value,
        sha256_file,
        tree_sha256,
        write_json,
    )
    from package_ice_wp10b_expert_review import (
        FORBIDDEN_PAYLOAD_PARTS,
        TEXT_SUFFIXES,
        _safe_reset,
        build_wp10b_review,
    )
except ModuleNotFoundError:
    from scripts.package_ice_wp10a_expert_review import (
        IDENTIFYING_TOKENS,
        LOCAL_TOKENS,
        ROOT,
        _iter_payload_files,
        file_record,
        read_json,
        sanitize_text,
        sanitize_value,
        sha256_file,
        tree_sha256,
        write_json,
    )
    from scripts.package_ice_wp10b_expert_review import (
        FORBIDDEN_PAYLOAD_PARTS,
        TEXT_SUFFIXES,
        _safe_reset,
        build_wp10b_review,
    )


DEFAULT_OUTPUT = ROOT / "outputs" / "sage_qdte_rce_c1_expert_review_v2_20260715"
PACKAGE_ID = "SAGE-QDTE-RCE-C1-EXPERT-REVIEW-20260715-v2"
MANIFEST_ID = "SAGE-QDTE-RCE-C1-EXPERT-REVIEW-MANIFEST-20260715-v2"

RCE_PATHS = (
    "docs/SAGE_QDTE_RCE_方法极限路线_20260715.md",
    "docs/SAGE_QDTE_RCE_C0_C1_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_RCE_C1_RESTRICTED_MIXTURE_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_RCE_C1_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md",
    "qdte/config_validation.py",
    "qdte/evolution/engine.py",
    "qdte/evolution/entropy.py",
    "qdte/evolution/precision.py",
    "qdte/evolution/rce_scoring.py",
    "qdte/evolution/transport.py",
    "qdte/measurement/public_artifact.py",
    "qdte/rce/__init__.py",
    "qdte/rce/confidence_set.py",
    "qdte/rce/controller.py",
    "qdte/rce/dual.py",
    "qdte/rce/integer.py",
    "qdte/rce/prior.py",
    "qdte/rce/relaxed.py",
    "scripts/analyze_rce_c1_postseal.py",
    "scripts/evaluate_rce_c1_panel.py",
    "scripts/evaluate_rce_c1_restricted_mixture.py",
    "scripts/generate_qdte_from_transcript.py",
    "scripts/package_rce_c1_expert_review.py",
    "scripts/run_rce_c1_cell.py",
    "scripts/run_rce_c1_panel.py",
    "scripts/run_rce_c1_restricted_mixture.py",
    "tests/test_analyze_rce_c1_postseal.py",
    "tests/test_evaluate_rce_c1_restricted_mixture.py",
    "tests/test_public_transcript_generation.py",
    "tests/test_rce_confidence_set.py",
    "tests/test_rce_dual.py",
    "tests/test_rce_integer.py",
    "tests/test_rce_relaxed.py",
    "tests/test_rce_scoring.py",
    "tests/test_run_rce_c1_restricted_mixture.py",
    "tests/test_package_rce_c1_expert_review.py",
)

RCE_RESULT_PATHS = (
    "outputs/sage_qdte_rce_c1_v3_20260715/panel_plan.json",
    "outputs/sage_qdte_rce_c1_v3_20260715/panel_status.json",
    "outputs/sage_qdte_rce_c1_v3_20260715/sealed_panel_manifest.json",
    "outputs/sage_qdte_rce_c1_v3_eval_20260715/summary.json",
    "outputs/sage_qdte_rce_c1_v3_eval_20260715/metrics.csv",
    "outputs/sage_qdte_rce_c1_v3_postseal_20260715/summary.json",
    "outputs/sage_qdte_rce_c1_v3_postseal_20260715/confidence_and_kl.csv",
    "outputs/sage_qdte_rce_c1_v3_postseal_20260715/metrics_with_initial.csv",
    "outputs/sage_qdte_rce_c1_restricted_mixture_v2_cpu_20260715/summary.json",
    "outputs/sage_qdte_rce_c1_restricted_mixture_v2_cpu_20260715/sealed_manifest.json",
    "outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/evaluation_plan.json",
    "outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/summary.json",
    "outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/metrics.csv",
    "outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/weights.csv",
)

DATASETS = ("adult", "br2000")
EPSILONS = ("0.1", "0.3")
SEEDS = (0, 1, 2)


def _cell_result_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for dataset in DATASETS:
        for epsilon in EPSILONS:
            for seed in SEEDS:
                root = (
                    f"outputs/sage_qdte_rce_c1_v3_20260715/{dataset}/"
                    f"epsilon_{epsilon}/seed_{seed}"
                )
                paths.extend(
                    (
                        f"{root}/mechanism_manifest.json",
                        f"{root}/cell_status.json",
                        f"{root}/generate_rce/metrics_final.json",
                        f"{root}/generate_rce/runtime.json",
                        f"{root}/generate_rce/run_status.json",
                    )
                )
    return tuple(paths)


def _restricted_cell_result_paths() -> tuple[str, ...]:
    return tuple(
        f"outputs/sage_qdte_rce_c1_restricted_mixture_v2_cpu_20260715/"
        f"{dataset}/epsilon_{epsilon}/seed_{seed}/restricted_result.json"
        for dataset in DATASETS
        for epsilon in EPSILONS
        for seed in SEEDS
    )


def _copy_sanitized_paths(
    source_root: Path,
    output: Path,
    paths: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relpath in paths:
        source = source_root / relpath
        if not source.is_file():
            raise FileNotFoundError(f"RCE C1 review payload is missing: {relpath}")
        destination = output / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        if destination.suffix == ".json":
            payload = sanitize_value(read_json(destination), source_root)
            payload["_expert_review_copy"] = {
                "local_paths_sanitized": True,
                "source_sha256": source_hash,
            }
            write_json(destination, payload)
        elif destination.suffix.lower() in TEXT_SUFFIXES or destination.suffix == ".py":
            text = sanitize_text(destination.read_text(encoding="utf-8"), source_root)
            for token in IDENTIFYING_TOKENS:
                text = text.replace(token, "anonymous-owner")
            destination.write_text(text, encoding="utf-8")
        records[relpath] = {
            "source_sha256": source_hash,
            "packaged_sha256": sha256_file(destination),
            "sanitized": source_hash != sha256_file(destination),
        }
    return records


def _readme() -> str:
    return """# SAGE-QDTE RCE C1 Expert Review Snapshot

Start with:

- [C1 result and decision questions](docs/SAGE_QDTE_RCE_C1_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md)
- [Frozen C0/C1 protocol](docs/SAGE_QDTE_RCE_C0_C1_PROTOCOL_20260715.md)
- [Expert RCE method-limit route](docs/SAGE_QDTE_RCE_方法极限路线_20260715.md)
- [Frozen offline evaluation](outputs/sage_qdte_rce_c1_v3_eval_20260715/summary.json)
- [Post-seal confidence/KL diagnosis](outputs/sage_qdte_rce_c1_v3_postseal_20260715/summary.json)
- [Restricted-mixture evaluation](outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/summary.json)
- [Restricted-mixture protocol](docs/SAGE_QDTE_RCE_C1_RESTRICTED_MIXTURE_PROTOCOL_20260715.md)
- [Sealed blind-panel manifest](outputs/sage_qdte_rce_c1_v3_20260715/sealed_panel_manifest.json)

The twelve-cell same-transcript panel was fully generated and sealed before
one frozen offline evaluation. RCE-v1 did not improve WP9 and did not beat
Official AIM. The post-seal diagnostic found truth inside all twelve confidence
sets, while RCE moved much closer to the released one-way product prior and
discarded useful interaction structure. The decision packet asks whether to
stop Adult rescue or authorize exactly one richer released structural prior.
The released-only restricted-mixture diagnostic removes interpolation within
the fixed initial/WP9/RCE convex hull. It improves integer RCE but still gives
RCE 82.8% mean weight, gives WP9 only 9.3%, and remains 21.2% worse than WP9.
This directly identifies minimum product-prior KL as the first bottleneck on
the declared convex hull, while making no global relaxed-ceiling claim.

This snapshot includes implementation, focused tests, sanitized manifests,
per-cell runtime/certificates, and aggregate true-utility metrics. It excludes
private rows, raw CSV, exact-answer caches, raw/released measurement payloads,
query payloads, initial tables, and synthetic tables. Original source hashes
are retained in `EXPERT_REVIEW_SOURCE_HASHES.json`.
"""


def build_rce_review(source_root: Path, output: Path, *, force: bool) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.resolve()
    _safe_reset(output, force=force)
    build_wp10b_review(source_root, output, force=False)
    (output / "EXPERT_REVIEW_MANIFEST.json").unlink()

    records = _copy_sanitized_paths(
        source_root,
        output,
        RCE_PATHS
        + RCE_RESULT_PATHS
        + _cell_result_paths()
        + _restricted_cell_result_paths(),
    )
    hashes_path = output / "EXPERT_REVIEW_SOURCE_HASHES.json"
    hashes = read_json(hashes_path)
    prior_records = hashes.get("records")
    if not isinstance(prior_records, dict):
        raise ValueError("Base expert-review source-hash records are invalid")
    prior_records.update(records)
    hashes["package_id"] = PACKAGE_ID
    write_json(hashes_path, hashes)
    (output / "EXPERT_REVIEW_README.md").write_text(_readme(), encoding="utf-8")

    payloads = _iter_payload_files(output)
    manifest_records = [file_record(path, output) for path in payloads]
    manifest = {
        "manifest_id": MANIFEST_ID,
        "package_id": PACKAGE_ID,
        "artifact_role": "sanitized_rce_c1_expert_review_snapshot",
        "private_rows_included": False,
        "released_transcript_included": False,
        "true_answer_caches_included": False,
        "synthetic_tables_included": False,
        "aggregate_offline_true_metrics_included": True,
        "payloads": manifest_records,
        "tree_sha256": tree_sha256(manifest_records),
    }
    write_json(output / "EXPERT_REVIEW_MANIFEST.json", manifest)
    errors = verify_rce_review(output)
    if errors:
        raise ValueError("RCE C1 expert-review verification failed: " + "; ".join(errors))
    return manifest


def verify_rce_review(output: Path) -> list[str]:
    output = output.resolve()
    manifest_path = output / "EXPERT_REVIEW_MANIFEST.json"
    if not manifest_path.is_file():
        return [f"Missing RCE C1 review manifest: {manifest_path}"]
    errors: list[str] = []
    manifest = read_json(manifest_path)
    if manifest.get("manifest_id") != MANIFEST_ID:
        errors.append("RCE C1 review manifest ID mismatch")
    records = manifest.get("payloads", [])
    expected = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    actual = {
        path.relative_to(output).as_posix(): path
        for path in _iter_payload_files(output)
    }
    if set(expected) != set(actual):
        errors.append(
            "RCE C1 payload mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    for relpath in sorted(set(expected) & set(actual)):
        path = actual[relpath]
        record = expected[relpath]
        if int(record.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"Byte-size mismatch: {relpath}")
        if str(record.get("sha256")) != sha256_file(path):
            errors.append(f"SHA-256 mismatch: {relpath}")
    canonical = [expected[path] for path in sorted(expected)]
    if manifest.get("tree_sha256") != tree_sha256(canonical):
        errors.append("RCE C1 review tree SHA-256 mismatch")

    for required in (
        RCE_PATHS
        + RCE_RESULT_PATHS
        + _cell_result_paths()
        + _restricted_cell_result_paths()
    ):
        if not (output / required).is_file():
            errors.append(f"Missing RCE C1 review payload: {required}")
    for relpath, path in actual.items():
        if any(token in relpath for token in FORBIDDEN_PAYLOAD_PARTS):
            errors.append(f"Forbidden raw/private payload included: {relpath}")
        if path.suffix.lower() in {".npy", ".npz"}:
            errors.append(f"Forbidden binary table/cache payload included: {relpath}")
        binary = path.read_bytes().lower()
        for token in IDENTIFYING_TOKENS:
            if token.encode("utf-8") in binary:
                errors.append(f"Identifying token {token!r} in {relpath}")
        # Packager/test source intentionally contains the literal path tokens it
        # detects. Runtime evidence and prose are the surfaces that may carry
        # machine-specific values and are scanned below, matching the existing
        # WP10a/WP10b review verifier.
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = binary.decode("utf-8", errors="replace")
        for token in LOCAL_TOKENS:
            if token in text:
                errors.append(f"Local path token {token!r} in {relpath}")

    seal = read_json(
        output / "outputs/sage_qdte_rce_c1_v3_20260715/sealed_panel_manifest.json"
    )
    if seal.get("num_cells") != 12 or seal.get("true_utility_evaluated") is not False:
        errors.append("RCE C1 blind-panel seal semantics changed")
    evaluation = read_json(
        output / "outputs/sage_qdte_rce_c1_v3_eval_20260715/summary.json"
    )
    classification = evaluation.get("classification", {})
    if any(classification.get(key) is not False for key in (
        "rce_improves_point_target_control",
        "adult_beats_aim_aggregate",
        "adult_beats_aim_at_both_epsilons",
    )):
        errors.append("RCE C1 negative-result classification changed")
    postseal = read_json(
        output / "outputs/sage_qdte_rce_c1_v3_postseal_20260715/summary.json"
    )
    truth = postseal.get("confidence", {}).get("truth", {})
    if truth.get("inside") != 12 or truth.get("total") != 12:
        errors.append("RCE C1 truth confidence-coverage diagnosis changed")
    restricted = read_json(
        output
        / "outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/summary.json"
    )
    restricted_classification = restricted.get("classification", {})
    if restricted_classification != {
        "beats_rce_v1_overall": True,
        "beats_wp9_overall": False,
        "product_kl_prefers_wp9_on_average": False,
    }:
        errors.append("Restricted-mixture estimator diagnosis changed")
    weights = restricted.get("mean_component_weights", {})
    if not (
        float(weights.get("rce_v1", 0.0)) > 0.8
        and float(weights.get("wp9_control", 1.0)) < 0.1
    ):
        errors.append("Restricted-mixture component preference changed")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify the sanitized RCE C1 review snapshot")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        errors = verify_rce_review(args.output_dir)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        manifest = read_json(args.output_dir / "EXPERT_REVIEW_MANIFEST.json")
        print(f"RCE C1 expert-review package verified: {args.output_dir.resolve()}")
        print(f"tree sha256: {manifest['tree_sha256']}")
        return 0
    manifest = build_rce_review(args.root, args.output_dir, force=args.force)
    print(f"RCE C1 expert-review package written: {args.output_dir.resolve()}")
    print(f"payloads: {len(manifest['payloads'])}")
    print(f"tree sha256: {manifest['tree_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
