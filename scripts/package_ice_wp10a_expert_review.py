#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

try:
    from path_defaults import repo_root
    from simulate_public_release import copy_release_surface
except ModuleNotFoundError:
    from scripts.path_defaults import repo_root
    from scripts.simulate_public_release import copy_release_surface


ROOT = repo_root()
DEFAULT_OUTPUT = ROOT / "outputs" / "sage_qdte_ice_wp10a_expert_review_20260715"

PACKAGE_ID = "SAGE-QDTE-ICE-WP10A-EXPERT-REVIEW-20260715-v1"
MANIFEST_ID = "SAGE-QDTE-ICE-WP10A-EXPERT-REVIEW-MANIFEST-20260715-v1"

DOCUMENT_PATHS = (
    "docs/SAGE_QDTE_ICE_WP9_STATIC_CONFIRMATION_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_ICE_WP9_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md",
    "docs/SAGE_QDTE_ICE_WP9A_PUBLIC_ROUTE_DIAGNOSTIC_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_ICE_WP9A_PUBLIC_ROUTE_DIAGNOSTIC_RESULT_20260715.md",
    "docs/SAGE_QDTE_ICE_WP10A_WORKLOAD_ALLOCATION_PROTOCOL_20260715.md",
    "docs/SAGE_QDTE_ICE_WP10A_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md",
)

SUPPORT_SCRIPT_PATHS = (
    "scripts/run_static_ice_measurement_pilot.py",
    "scripts/run_static_ice_qdte_pilot.py",
    "scripts/run_coverage_refinement_wp8a.py",
    "scripts/run_sage_voi_selector_pilot.py",
    "scripts/run_sage_voi_selector_panel.py",
    "scripts/evaluate_sage_voi_selector_panel.py",
    "scripts/run_static_ice_wp9_cell.py",
    "scripts/run_static_ice_wp9_panel.py",
    "scripts/evaluate_static_ice_wp9_panel.py",
    "scripts/plot_static_ice_wp9_gate.py",
    "scripts/audit_static_ice_wp9_public_routes.py",
    "scripts/run_static_ice_wp10a_workload_allocation.py",
    "scripts/evaluate_static_ice_wp10a_workload_allocation.py",
    "scripts/package_ice_wp10a_expert_review.py",
)

SUPPORT_TEST_PATHS = (
    "tests/test_orthogonal_interactions.py",
    "tests/test_interaction_factorization.py",
    "tests/test_interaction_adapter.py",
    "tests/test_workload_factorization.py",
    "tests/test_run_static_ice_wp9_cell.py",
    "tests/test_run_static_ice_wp9_panel.py",
    "tests/test_evaluate_static_ice_wp9_panel.py",
    "tests/test_plot_static_ice_wp9_gate.py",
    "tests/test_audit_static_ice_wp9_public_routes.py",
    "tests/test_run_static_ice_wp10a_workload_allocation.py",
    "tests/test_evaluate_static_ice_wp10a_workload_allocation.py",
    "tests/test_package_ice_wp10a_expert_review.py",
)

SANITIZED_SOURCE_PATHS = (
    "scripts/evaluate_static_ice_wp9_panel.py",
    "scripts/run_sage_voi_selector_panel.py",
    "scripts/run_static_ice_wp9_panel.py",
    "tests/test_run_static_ice_wp9_panel.py",
)

EVIDENCE_PATHS = (
    "outputs/static_ice_wp9_formal_20260715/panel_plan.json",
    "outputs/static_ice_wp9_formal_20260715/sealed_panel_manifest.json",
    "outputs/static_ice_wp9_eval_20260715/gate_summary.json",
    "outputs/static_ice_wp9_eval_20260715/metrics.csv",
    "outputs/static_ice_wp9_figures_20260715/figure_manifest.json",
    "outputs/static_ice_wp9_figures_20260715/wp9_static_ice_vs_aim_gate.pdf",
    "outputs/static_ice_wp9_figures_20260715/wp9_static_ice_vs_aim_gate.png",
    "outputs/static_ice_wp9_figures_20260715/wp9_static_ice_vs_aim_ratios.csv",
    "outputs/static_ice_wp9a_public_routes_formal_20260715/manifest.json",
    "outputs/static_ice_wp9a_public_routes_formal_20260715/public_route_diagnostics.json",
    "outputs/static_ice_wp9a_public_routes_formal_20260715/allocation_risk_summary.csv",
    "outputs/static_ice_wp9a_public_routes_formal_20260715/released_tree_stability.csv",
    "outputs/static_ice_wp10a_workload_l2_formal_20260715/plan.json",
    "outputs/static_ice_wp10a_workload_l2_formal_20260715/sealed_manifest.json",
    "outputs/static_ice_wp10a_workload_l2_formal_20260715/wp10a_status.json",
    "outputs/static_ice_wp10a_workload_l2_eval_20260715/manifest.json",
    "outputs/static_ice_wp10a_workload_l2_eval_20260715/gate_summary.json",
)

PUBLIC_INPUT_DATASETS = (
    "nltcs_sage_strong",
    "acs_sage_strong",
    "br2000_sage_strong",
    "adult_sage_strong",
)
PUBLIC_INPUT_FILENAMES = ("schema.json", "metadata.json")

JSON_SUFFIXES = {".json"}
TEXT_SUFFIXES = {".csv", ".ini", ".json", ".md", ".txt", ".yaml", ".yml"}
SOURCE_SUFFIXES = {".py"}
IDENTIFYING_TOKENS = ("qian" + "qiu", "houyu" + "wushang", "train" + "34")
LOCAL_TOKENS = ("/home/", "/mnt/", "file://", "vscode://")
FORBIDDEN_PAYLOAD_PARTS = (
    "offline_true_answer_cache",
    "synthetic_encoded.npy",
    "synthetic_initial_encoded.npy",
    "measurements.json",
    "queries.json",
)
MARKDOWN_OUTPUT_LINK = re.compile(r"\]\((\.\./outputs/[^)]+)\)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def tree_sha256(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_reset_output(output: Path, *, force: bool) -> None:
    resolved = output.resolve()
    if resolved.exists() and not force:
        raise FileExistsError(f"Output already exists: {resolved}")
    if resolved == ROOT.resolve() or resolved == Path(resolved.anchor) or len(resolved.parts) < 4:
        raise ValueError(f"Refusing unsafe expert-review output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _external_input_root(source_root: Path) -> Path:
    packaged = source_root / "external_inputs"
    if packaged.is_dir():
        return packaged
    return source_root.parent / "baseline" / "private-de" / "external_inputs"


def sanitize_string(value: str, source_root: Path) -> str:
    source_base = str(source_root.resolve())
    external_base = str(_external_input_root(source_root).resolve())
    source_prefix = source_base + "/"
    external_prefix = external_base + "/"
    if value == source_base:
        return "."
    if value == external_base:
        return "external_inputs"
    if value.startswith(source_prefix):
        return value[len(source_prefix) :]
    if value.startswith(external_prefix):
        return "external_inputs/" + value[len(external_prefix) :]
    if value.startswith("/home/") or value.startswith("/mnt/"):
        name = Path(value).name
        return f"external_artifacts/{name}" if name else "external_artifacts"
    return value.replace(source_prefix, "").replace(external_prefix, "external_inputs/")


def sanitize_value(value: Any, source_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item, source_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, source_root) for item in value]
    if isinstance(value, str):
        return sanitize_string(value, source_root)
    return value


def sanitize_text(text: str, source_root: Path) -> str:
    source_base = str(source_root.resolve())
    external_base = str(_external_input_root(source_root).resolve())
    source_prefix = source_base + "/"
    external_prefix = external_base + "/"
    text = text.replace(source_prefix, "")
    text = text.replace(external_prefix, "external_inputs/")
    text = text.replace(external_base, "external_inputs")
    text = text.replace(source_base, ".")
    text = re.sub(r"/(?:home|mnt)/[^\s`\"')\]]+", "<local-path-omitted>", text)
    return text


def _copy_overlay(source_root: Path, output: Path, paths: Iterable[str]) -> None:
    for relpath in paths:
        source = source_root / relpath
        if not source.is_file():
            raise FileNotFoundError(f"Expert-review source path is missing: {relpath}")
        destination = output / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_public_inputs(source_root: Path, output: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    external_root = _external_input_root(source_root)
    for dataset in PUBLIC_INPUT_DATASETS:
        for filename in PUBLIC_INPUT_FILENAMES:
            source = external_root / dataset / filename
            if not source.is_file():
                raise FileNotFoundError(f"Public expert-review input is missing: {source}")
            relpath = f"external_inputs/{dataset}/{filename}"
            destination = output / relpath
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = sanitize_value(read_json(source), source_root)
            payload["_expert_review_copy"] = {
                "local_paths_sanitized": True,
                "source_sha256": sha256_file(source),
            }
            write_json(destination, payload)
            records[relpath] = {
                "source_sha256": sha256_file(source),
                "packaged_sha256": sha256_file(destination),
                "sanitized": True,
            }
    return records


def _sanitize_selected_payloads(
    source_root: Path,
    output: Path,
    paths: Iterable[str],
) -> dict[str, dict[str, Any]]:
    source_hashes: dict[str, dict[str, Any]] = {}
    for relpath in paths:
        source = source_root / relpath
        packaged = output / relpath
        source_hash = sha256_file(source)
        if packaged.suffix.lower() in JSON_SUFFIXES:
            payload = sanitize_value(read_json(packaged), source_root)
            if isinstance(payload, dict):
                payload["_expert_review_copy"] = {
                    "local_paths_sanitized": True,
                    "source_sha256": source_hash,
                }
            write_json(packaged, payload)
        elif packaged.suffix.lower() in {".md", ".py", ".txt", ".yaml", ".yml", ".csv"}:
            text = packaged.read_text(encoding="utf-8")
            packaged.write_text(sanitize_text(text, source_root), encoding="utf-8")
        source_hashes[relpath] = {
            "source_sha256": source_hash,
            "packaged_sha256": sha256_file(packaged),
            "sanitized": source_hash != sha256_file(packaged),
        }
    return source_hashes


def _review_readme() -> str:
    return """# SAGE-QDTE-ICE WP9/WP10a Expert Review Snapshot

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
"""


def build_review_package(source_root: Path, output: Path, *, force: bool) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.resolve()
    _safe_reset_output(output, force=force)
    copy_release_surface(source_root, output, force=True)

    overlays = (*DOCUMENT_PATHS, *SUPPORT_SCRIPT_PATHS, *SUPPORT_TEST_PATHS, *EVIDENCE_PATHS)
    _copy_overlay(source_root, output, overlays)
    source_hashes = _sanitize_selected_payloads(
        source_root,
        output,
        (*DOCUMENT_PATHS, *EVIDENCE_PATHS, *SANITIZED_SOURCE_PATHS),
    )
    source_hashes.update(_copy_public_inputs(source_root, output))

    (output / "EXPERT_REVIEW_README.md").write_text(_review_readme(), encoding="utf-8")
    write_json(
        output / "EXPERT_REVIEW_SOURCE_HASHES.json",
        {
            "package_id": PACKAGE_ID,
            "note": "Original local hashes before repository-path sanitization.",
            "records": source_hashes,
        },
    )

    payloads = _iter_payload_files(output)
    records = [file_record(path, output) for path in payloads]
    manifest = {
        "manifest_id": MANIFEST_ID,
        "package_id": PACKAGE_ID,
        "artifact_role": "sanitized_expert_review_snapshot",
        "private_rows_included": False,
        "true_answer_caches_included": False,
        "raw_measurement_query_payloads_included": False,
        "payloads": records,
        "tree_sha256": tree_sha256(records),
    }
    write_json(output / "EXPERT_REVIEW_MANIFEST.json", manifest)
    errors = verify_review_package(output)
    if errors:
        raise ValueError("Expert-review package verification failed: " + "; ".join(errors))
    return manifest


def _iter_payload_files(output: Path) -> list[Path]:
    files: list[Path] = []
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix == ".pyc":
            continue
        if path.is_file() and path.name != "EXPERT_REVIEW_MANIFEST.json":
            files.append(path)
    return sorted(files)


def verify_review_package(output: Path) -> list[str]:
    output = output.resolve()
    manifest_path = output / "EXPERT_REVIEW_MANIFEST.json"
    if not manifest_path.is_file():
        return [f"Missing expert-review manifest: {manifest_path}"]
    errors: list[str] = []
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"Invalid expert-review manifest: {exc}"]
    if manifest.get("manifest_id") != MANIFEST_ID:
        errors.append("Expert-review manifest ID mismatch")

    records = manifest.get("payloads", [])
    expected = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    actual = {path.relative_to(output).as_posix(): path for path in _iter_payload_files(output)}
    if set(expected) != set(actual):
        errors.append(
            "Payload path mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    for relpath in sorted(set(expected) & set(actual)):
        record = expected[relpath]
        path = actual[relpath]
        if int(record.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"Byte-size mismatch: {relpath}")
        if str(record.get("sha256")) != sha256_file(path):
            errors.append(f"SHA-256 mismatch: {relpath}")
    canonical = [expected[path] for path in sorted(expected)]
    if manifest.get("tree_sha256") != tree_sha256(canonical):
        errors.append("Expert-review tree SHA-256 mismatch")

    for required in (*DOCUMENT_PATHS, *EVIDENCE_PATHS, "EXPERT_REVIEW_README.md"):
        if not (output / required).is_file():
            errors.append(f"Required expert-review payload is missing: {required}")

    for path in actual.values():
        relative = path.relative_to(output).as_posix()
        if any(part in relative for part in FORBIDDEN_PAYLOAD_PARTS):
            errors.append(f"Forbidden raw/private payload included: {relative}")
        binary = path.read_bytes().lower()
        for token in IDENTIFYING_TOKENS:
            if token.encode("utf-8") in binary:
                errors.append(f"Identifying token {token!r} in {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = binary.decode("utf-8", errors="replace")
        for token in LOCAL_TOKENS:
            if token in text:
                errors.append(f"Local path token {token!r} in {relative}")

    for relpath in DOCUMENT_PATHS:
        doc = output / relpath
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8")
        for match in MARKDOWN_OUTPUT_LINK.findall(text):
            target = (doc.parent / match).resolve()
            try:
                target.relative_to(output)
            except ValueError:
                errors.append(f"Markdown output link escapes package: {relpath} -> {match}")
                continue
            if not target.is_file():
                errors.append(f"Broken Markdown output link: {relpath} -> {match}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify the ICE WP10a expert-review branch snapshot.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        errors = verify_review_package(args.output_dir)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        manifest = read_json(args.output_dir / "EXPERT_REVIEW_MANIFEST.json")
        print(f"expert-review package verified: {args.output_dir.resolve()}")
        print(f"tree sha256: {manifest['tree_sha256']}")
        return 0
    manifest = build_review_package(args.root, args.output_dir, force=args.force)
    print(f"expert-review package written: {args.output_dir.resolve()}")
    print(f"payloads: {len(manifest['payloads'])}")
    print(f"tree sha256: {manifest['tree_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
