#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from package_ice_wp10a_expert_review import (
        IDENTIFYING_TOKENS,
        LOCAL_TOKENS,
        ROOT,
        _iter_payload_files,
        build_review_package,
        file_record,
        read_json,
        sanitize_text,
        sanitize_value,
        sha256_file,
        tree_sha256,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.package_ice_wp10a_expert_review import (
        IDENTIFYING_TOKENS,
        LOCAL_TOKENS,
        ROOT,
        _iter_payload_files,
        build_review_package,
        file_record,
        read_json,
        sanitize_text,
        sanitize_value,
        sha256_file,
        tree_sha256,
        write_json,
    )


DEFAULT_OUTPUT = ROOT / "outputs" / "sage_qdte_ice_wp10b_expert_review_20260715"
PACKAGE_ID = "SAGE-QDTE-ICE-WP10B-GCEA-EXPERT-REVIEW-20260715-v1"
MANIFEST_ID = "SAGE-QDTE-ICE-WP10B-GCEA-EXPERT-REVIEW-MANIFEST-20260715-v1"

WP10B_PATHS = (
    "docs/SAGE_QDTE_ICE_WP10B_GCEA_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_ICE_WP10B_GCEA_PUBLIC_GATE_RESULT_20260715.md",
    "qdte/measurement/gcea.py",
    "scripts/run_static_ice_wp10b_gcea_public_gate.py",
    "scripts/package_ice_wp10b_expert_review.py",
    "tests/test_gcea.py",
    "tests/test_run_static_ice_wp10b_gcea_public_gate.py",
    "tests/test_package_ice_wp10b_expert_review.py",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/plan.json",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.json",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.csv",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/sealed_manifest.json",
)

SANITIZED_PATHS = (
    "docs/SAGE_QDTE_ICE_WP10B_GCEA_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_ICE_WP10B_GCEA_PUBLIC_GATE_RESULT_20260715.md",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/plan.json",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.json",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.csv",
    "outputs/static_ice_wp10b_gcea_public_gate_20260715/sealed_manifest.json",
)

TEXT_SUFFIXES = {".csv", ".ini", ".json", ".md", ".txt", ".yaml", ".yml"}
FORBIDDEN_PAYLOAD_PARTS = (
    "offline_true_answer_cache",
    "real_encoded",
    "synthetic_encoded.npy",
    "synthetic_initial_encoded.npy",
    "measurements.json",
    "queries.json",
)
PUBLIC_GATE_DATASETS = ("adult", "br2000")
PUBLIC_GATE_INPUT_NAMES = (
    "schema.json",
    "metadata.json",
    "queries_full.json",
    "workload_groups.json",
)


def _safe_reset(output: Path, *, force: bool) -> None:
    resolved = output.resolve()
    if resolved.exists() and not force:
        raise FileExistsError(f"Output already exists: {resolved}")
    if resolved == ROOT.resolve() or resolved == Path(resolved.anchor) or len(resolved.parts) < 4:
        raise ValueError(f"Refusing unsafe WP10b review output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _copy_wp10b_overlay(source_root: Path, output: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relpath in WP10B_PATHS:
        source = source_root / relpath
        if not source.is_file():
            raise FileNotFoundError(f"WP10b review payload is missing: {relpath}")
        destination = output / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        if relpath in SANITIZED_PATHS:
            if destination.suffix == ".json":
                payload = sanitize_value(read_json(destination), source_root)
                payload["_expert_review_copy"] = {
                    "local_paths_sanitized": True,
                    "source_sha256": source_hash,
                }
                write_json(destination, payload)
            else:
                text = destination.read_text(encoding="utf-8")
                destination.write_text(sanitize_text(text, source_root), encoding="utf-8")
        records[relpath] = {
            "source_sha256": source_hash,
            "packaged_sha256": sha256_file(destination),
            "sanitized": source_hash != sha256_file(destination),
        }
    return records


def _copy_public_gate_inputs(source_root: Path, output: Path) -> dict[str, dict[str, Any]]:
    packaged_input_root = source_root / "external_inputs"
    input_root = (
        packaged_input_root
        if (packaged_input_root / "adult_sage_strong" / "queries_full.json").is_file()
        else source_root.parent / "baseline" / "private-de" / "external_inputs"
    )
    records: dict[str, dict[str, Any]] = {}
    for dataset in PUBLIC_GATE_DATASETS:
        for name in PUBLIC_GATE_INPUT_NAMES:
            source = input_root / f"{dataset}_sage_strong" / name
            if not source.is_file():
                raise FileNotFoundError(f"WP10b public-gate input is missing: {source}")
            relpath = Path("external_inputs") / f"{dataset}_sage_strong" / name
            destination = output / relpath
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records[relpath.as_posix()] = {
                "source_sha256": sha256_file(source),
                "packaged_sha256": sha256_file(destination),
                "sanitized": False,
            }
    return records


def _readme() -> str:
    return """# SAGE-QDTE-ICE WP10b GCEA Expert Review Snapshot

Start with:

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
"""


def build_wp10b_review(source_root: Path, output: Path, *, force: bool) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.resolve()
    _safe_reset(output, force=force)
    build_review_package(source_root, output, force=False)
    (output / "EXPERT_REVIEW_MANIFEST.json").unlink()

    new_hashes = _copy_wp10b_overlay(source_root, output)
    new_hashes.update(_copy_public_gate_inputs(source_root, output))
    hashes_path = output / "EXPERT_REVIEW_SOURCE_HASHES.json"
    hashes = read_json(hashes_path)
    records = hashes.get("records")
    if not isinstance(records, dict):
        raise ValueError("Base expert-review source-hash records are invalid")
    records.update(new_hashes)
    hashes["package_id"] = PACKAGE_ID
    write_json(hashes_path, hashes)
    (output / "EXPERT_REVIEW_README.md").write_text(_readme(), encoding="utf-8")

    payloads = _iter_payload_files(output)
    manifest_records = [file_record(path, output) for path in payloads]
    manifest = {
        "manifest_id": MANIFEST_ID,
        "package_id": PACKAGE_ID,
        "artifact_role": "sanitized_wp10b_public_gate_expert_review_snapshot",
        "private_rows_included": False,
        "released_transcript_included": False,
        "true_answer_caches_included": False,
        "synthetic_tables_included": False,
        "payloads": manifest_records,
        "tree_sha256": tree_sha256(manifest_records),
    }
    write_json(output / "EXPERT_REVIEW_MANIFEST.json", manifest)
    errors = verify_wp10b_review(output)
    if errors:
        raise ValueError("WP10b expert-review verification failed: " + "; ".join(errors))
    return manifest


def verify_wp10b_review(output: Path) -> list[str]:
    output = output.resolve()
    manifest_path = output / "EXPERT_REVIEW_MANIFEST.json"
    if not manifest_path.is_file():
        return [f"Missing WP10b review manifest: {manifest_path}"]
    errors: list[str] = []
    manifest = read_json(manifest_path)
    if manifest.get("manifest_id") != MANIFEST_ID:
        errors.append("WP10b review manifest ID mismatch")
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
            "WP10b payload mismatch: "
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
        errors.append("WP10b review tree SHA-256 mismatch")

    for required in WP10B_PATHS:
        if not (output / required).is_file():
            errors.append(f"Missing WP10b review payload: {required}")
    for dataset in PUBLIC_GATE_DATASETS:
        for name in PUBLIC_GATE_INPUT_NAMES:
            required = Path("external_inputs") / f"{dataset}_sage_strong" / name
            if not (output / required).is_file():
                errors.append(f"Missing WP10b public-gate input: {required.as_posix()}")
    for relpath, path in actual.items():
        if any(token in relpath for token in FORBIDDEN_PAYLOAD_PARTS):
            errors.append(f"Forbidden raw/private payload included: {relpath}")
        binary = path.read_bytes().lower()
        for token in IDENTIFYING_TOKENS:
            if token.encode("utf-8") in binary:
                errors.append(f"Identifying token {token!r} in {relpath}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = binary.decode("utf-8", errors="replace")
        for token in LOCAL_TOKENS:
            if token in text:
                errors.append(f"Local path token {token!r} in {relpath}")
    gate = read_json(output / "outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.json")
    if gate.get("generation_authorized") is not False:
        errors.append("WP10b review gate must preserve generation_authorized=false")
    if gate.get("decision") != "q1_a_freeze_ice_binary_profile":
        errors.append("WP10b review gate decision mismatch")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify the sanitized WP10b GCEA review snapshot.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        errors = verify_wp10b_review(args.output_dir)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        manifest = read_json(args.output_dir / "EXPERT_REVIEW_MANIFEST.json")
        print(f"WP10b expert-review package verified: {args.output_dir.resolve()}")
        print(f"tree sha256: {manifest['tree_sha256']}")
        return 0
    manifest = build_wp10b_review(args.root, args.output_dir, force=args.force)
    print(f"WP10b expert-review package written: {args.output_dir.resolve()}")
    print(f"payloads: {len(manifest['payloads'])}")
    print(f"tree sha256: {manifest['tree_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
