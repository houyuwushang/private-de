from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdte.config_validation import validate_config
from qdte.dataio import read_json, write_json
from qdte.evolution.initialization import split_run_rng_streams
from qdte.measurement.measure import Measurements, measurements_from_public_dict
from qdte.measurement.measure import measure_real_dataset
from qdte.preprocess import load_and_preprocess_csv
from qdte.privacy.accountant import zcdp_epsilon
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import build_workload
from qdte.schema import TableSchema


PUBLIC_TRANSCRIPT_PROTOCOL = "qdte-public-measurement-transcript-v1"
PUBLIC_TRANSCRIPT_MANIFEST = "transcript_manifest.json"
PUBLIC_TRANSCRIPT_FILES = {
    "schema": "schema.json",
    "queries": "queries.json",
    "measurements": "measurements.json",
}


@dataclass(frozen=True)
class VerifiedPublicTranscript:
    root: Path
    manifest: dict[str, Any]
    schema: TableSchema
    queries: QueryCatalogue
    measurements: Measurements


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Public transcript is missing {relative_path}: {path}")
    if path.is_symlink():
        raise ValueError(f"Public transcript artifacts must not be symbolic links: {path}")
    return {
        "path": relative_path,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _verify_record(root: Path, expected_path: str, record: dict[str, Any]) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Public transcript record {expected_path!r} must be a mapping")
    if record.get("path") != expected_path:
        raise ValueError(
            f"Public transcript record path must be {expected_path!r}, got {record.get('path')!r}"
        )
    observed = _artifact_record(root, expected_path)
    if observed != record:
        raise RuntimeError(f"Public transcript artifact changed after sealing: {expected_path}")
    return root / expected_path


def _validate_privacy_ledger(
    measurements: Measurements,
    privacy_ledger_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if measurements.mode != "dp":
        raise ValueError("Public transcript generation requires a DP measurement artifact")
    if measurements.num_rows is None or int(measurements.num_rows) <= 0:
        raise ValueError("Public transcript must declare a positive public row count")
    embedded_ledger = measurements.privacy_ledger
    if embedded_ledger is not None and privacy_ledger_override is not None:
        if embedded_ledger != privacy_ledger_override:
            raise ValueError(
                "Public transcript manifest ledger does not match the embedded ledger"
            )
    ledger = embedded_ledger or privacy_ledger_override
    if not isinstance(ledger, dict):
        raise ValueError("Public transcript requires a serialized privacy_ledger")
    if ledger.get("accounting") != "zcdp_actual_spend_v1":
        raise ValueError("Public transcript requires zcdp_actual_spend_v1 accounting")
    if ledger.get("adjacency") != "add_remove":
        raise ValueError("Public transcript requires add_remove adjacency")

    rho_limit = float(ledger.get("rho_limit", float("nan")))
    rho_spent = float(ledger.get("rho_spent", float("nan")))
    delta = float(ledger.get("delta", float("nan")))
    epsilon = float(ledger.get("epsilon_from_actual_spend", float("nan")))
    values = (rho_limit, rho_spent, delta, epsilon)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Public transcript privacy ledger contains non-finite totals")
    if rho_limit <= 0.0 or rho_spent <= 0.0 or rho_spent > rho_limit + 1.0e-12 * max(1.0, rho_limit):
        raise ValueError("Public transcript privacy ledger has invalid rho totals")
    if not 0.0 < delta < 1.0:
        raise ValueError("Public transcript privacy ledger has invalid delta")
    if not math.isclose(rho_limit, measurements.rho_total, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("Public transcript rho_limit does not match measurements.rho_total")
    if not math.isclose(rho_spent, measurements.rho_spent, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("Public transcript ledger spend does not match measurements.rho_spent")
    if not math.isclose(delta, measurements.delta, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Public transcript ledger delta does not match measurements.delta")
    expected_epsilon = zcdp_epsilon(rho_spent, delta)
    if not math.isclose(epsilon, expected_epsilon, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("Public transcript epsilon is not computed from actual rho spend")
    if not math.isclose(
        measurements.epsilon_delta,
        expected_epsilon,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Measurement epsilon_delta is not computed from actual rho spend")

    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Public transcript privacy ledger must contain mechanism entries")
    entry_rhos: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Privacy ledger entry {index} must be a mapping")
        if not str(entry.get("label", "")) or not str(entry.get("mechanism", "")):
            raise ValueError(f"Privacy ledger entry {index} is missing label or mechanism")
        entry_rho = float(entry.get("rho", float("nan")))
        if not math.isfinite(entry_rho) or entry_rho <= 0.0:
            raise ValueError(f"Privacy ledger entry {index} has invalid rho")
        metadata = entry.get("public_metadata")
        if not isinstance(metadata, dict) or metadata.get("adjacency") != "add_remove":
            raise ValueError(f"Privacy ledger entry {index} is missing add_remove provenance")
        entry_rhos.append(entry_rho)
    if not math.isclose(math.fsum(entry_rhos), rho_spent, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("Privacy ledger entry charges do not sum to actual rho spend")
    return ledger


def _load_payloads(
    root: Path,
    privacy_ledger_override: dict[str, Any] | None = None,
) -> tuple[TableSchema, QueryCatalogue, Measurements, dict[str, Any]]:
    schema = TableSchema.load_json(root / PUBLIC_TRANSCRIPT_FILES["schema"])
    queries = QueryCatalogue.from_dict(read_json(root / PUBLIC_TRANSCRIPT_FILES["queries"]))
    measurements = measurements_from_public_dict(
        read_json(root / PUBLIC_TRANSCRIPT_FILES["measurements"])
    )
    schema.validate()
    queries.validate(schema.cardinalities)
    if measurements.target_projected.shape != (queries.m,):
        raise ValueError("Public transcript target length does not match queries.json")
    ledger = _validate_privacy_ledger(measurements, privacy_ledger_override)
    return schema, queries, measurements, ledger


def write_public_transcript_manifest(
    root: str | Path,
    *,
    privacy_ledger_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = Path(root).resolve()
    schema, queries, measurements, ledger = _load_payloads(
        artifact_root,
        privacy_ledger_override,
    )
    artifacts = {
        name: _artifact_record(artifact_root, relative_path)
        for name, relative_path in PUBLIC_TRANSCRIPT_FILES.items()
    }
    manifest = {
        "protocol_id": PUBLIC_TRANSCRIPT_PROTOCOL,
        "artifact_role": "public_dp_measurement_transcript",
        "private_input_hashes_excluded": True,
        "true_utility_evaluated": False,
        "generation_authorized": True,
        "public_n_rows": int(measurements.num_rows),
        "num_columns": int(schema.d),
        "num_queries": int(queries.m),
        "privacy": {
            "mode": "dp",
            "adjacency": "add_remove",
            "rho_total": float(measurements.rho_total),
            "rho_spent": float(measurements.rho_spent),
            "delta": float(measurements.delta),
            "epsilon_from_actual_spend": float(measurements.epsilon_delta),
            "accounting": str(ledger["accounting"]),
        },
        "artifacts": artifacts,
    }
    if measurements.privacy_ledger is None:
        manifest["privacy_ledger_override"] = ledger
    write_json(manifest, artifact_root / PUBLIC_TRANSCRIPT_MANIFEST)
    return manifest


def verify_public_transcript(root: str | Path) -> VerifiedPublicTranscript:
    artifact_root = Path(root).resolve()
    manifest_path = artifact_root / PUBLIC_TRANSCRIPT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Public transcript is missing {PUBLIC_TRANSCRIPT_MANIFEST}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Public transcript manifest must be a mapping")
    expected = {
        "protocol_id": PUBLIC_TRANSCRIPT_PROTOCOL,
        "artifact_role": "public_dp_measurement_transcript",
        "private_input_hashes_excluded": True,
        "true_utility_evaluated": False,
        "generation_authorized": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Public transcript manifest violates {key}: {manifest.get(key)!r}")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != set(PUBLIC_TRANSCRIPT_FILES):
        raise ValueError("Public transcript manifest must seal exactly schema, queries, and measurements")
    for name, relative_path in PUBLIC_TRANSCRIPT_FILES.items():
        _verify_record(artifact_root, relative_path, raw_artifacts[name])

    raw_ledger_override = manifest.get("privacy_ledger_override")
    if raw_ledger_override is not None and not isinstance(raw_ledger_override, dict):
        raise ValueError("Public transcript privacy_ledger_override must be a mapping")
    schema, queries, measurements, ledger = _load_payloads(
        artifact_root,
        raw_ledger_override,
    )
    if measurements.privacy_ledger is not None and raw_ledger_override is not None:
        raise ValueError(
            "Public transcript must not duplicate an embedded privacy ledger in its manifest"
        )
    if int(manifest.get("public_n_rows", 0)) != int(measurements.num_rows):
        raise ValueError("Public transcript manifest row count does not match measurements")
    if int(manifest.get("num_columns", 0)) != schema.d:
        raise ValueError("Public transcript manifest column count does not match schema")
    if int(manifest.get("num_queries", 0)) != queries.m:
        raise ValueError("Public transcript manifest query count does not match queries")
    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("Public transcript manifest is missing privacy metadata")
    comparisons = {
        "mode": "dp",
        "adjacency": "add_remove",
        "accounting": ledger["accounting"],
    }
    for key, value in comparisons.items():
        if privacy.get(key) != value:
            raise ValueError(f"Public transcript manifest privacy.{key} mismatch")
    numeric = {
        "rho_total": measurements.rho_total,
        "rho_spent": measurements.rho_spent,
        "delta": measurements.delta,
        "epsilon_from_actual_spend": measurements.epsilon_delta,
    }
    for key, value in numeric.items():
        if not math.isclose(
            float(privacy.get(key, float("nan"))),
            float(value),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Public transcript manifest privacy.{key} mismatch")
    return VerifiedPublicTranscript(
        root=artifact_root,
        manifest=manifest,
        schema=schema,
        queries=queries,
        measurements=measurements,
    )


def materialize_public_transcript(
    config: dict[str, Any],
    root: str | Path,
) -> VerifiedPublicTranscript:
    """Run only the private DP measurement stage and seal its public output."""
    validate_config(config)
    run_cfg = config.get("run", {})
    privacy_cfg = config.get("privacy", {})
    measurement_cfg = config.get("measurement", {}) or {}
    workload_cfg = config.get("workload", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    if not bool(privacy_cfg.get("dp_release_mode", False)):
        raise ValueError("Public transcript materialization requires privacy.dp_release_mode=true")
    if bool(run_cfg.get("transcript_only_generation", False)):
        raise ValueError("Measurement materialization cannot run in transcript-only generation mode")
    if measurement_cfg.get("reuse_from", measurement_cfg.get("artifact_dir")) not in {None, ""}:
        raise ValueError("Measurement materialization cannot reuse an existing measurement artifact")
    if bool(workload_cfg.get("reuse_from_measurement", False)):
        raise ValueError("Measurement materialization must construct its workload before release")

    artifact_root = Path(root).resolve()
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(
            f"Public transcript output directory must be absent or empty: {artifact_root}"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)

    preprocess_result = load_and_preprocess_csv(config)
    X_real = preprocess_result.X
    schema = preprocess_result.schema
    public_n_rows = int(privacy_cfg["public_n_rows"])
    if int(X_real.shape[0]) != public_n_rows:
        raise ValueError(
            "Private input row count does not match declared privacy.public_n_rows: "
            f"input={X_real.shape[0]}, public={public_n_rows}"
        )
    qcat, workload_groups = build_workload(schema, config)
    if qcat.m <= 0:
        raise ValueError("Configured workload produced no queries")
    schema.validate()
    qcat.validate(schema.cardinalities)
    measurement_rng, _ = split_run_rng_streams(int(run_cfg.get("seed", 0)))
    measurements = measure_real_dataset(
        X_real,
        qcat,
        workload_groups,
        config,
        measurement_rng,
        batch_size=int(runtime_cfg.get("answer_batch_size", 8192)),
        cardinalities=schema.cardinalities,
    )
    if measurements.mode != "dp":
        raise ValueError("Public transcript materialization produced a non-DP artifact")
    if measurements.num_rows != public_n_rows:
        raise ValueError("Public transcript materialization lost the declared public row count")

    schema.save_json(artifact_root / PUBLIC_TRANSCRIPT_FILES["schema"])
    qcat.save_json(artifact_root / PUBLIC_TRANSCRIPT_FILES["queries"])
    write_json(
        measurements.to_public_dict(),
        artifact_root / PUBLIC_TRANSCRIPT_FILES["measurements"],
    )
    write_public_transcript_manifest(artifact_root)
    return verify_public_transcript(artifact_root)
