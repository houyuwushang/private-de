#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs


DATASETS = ("adult_sage_strong", "acs_sage_strong", "br2000_sage_strong", "nltcs_sage_strong")
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class MethodSpec:
    method: str
    root: str
    gpu_policy: str
    expected_env: str
    pattern_template: str
    expected_label: str
    note: str


METHODS = (
    MethodSpec(
        method="sage",
        root="sage",
        gpu_policy="required",
        expected_env="qdte",
        pattern_template="{dataset}/rho1p0/seed{seed}/run_metadata.json",
        expected_label="SAGE/QDTE",
        note="SAGE should use the GPU dense scoring backend for the paper runs.",
    ),
    MethodSpec(
        method="private_gsd_gpu_1m_fulln_audit",
        root="private_gsd_gpu_1m_fulln_audit",
        gpu_policy="required",
        expected_env="gsd",
        pattern_template="{dataset}/rho1p0/seed{seed}/run_metadata.json",
        expected_label="Private-GSD GPU 1M/full-N",
        note="High-power Private-GSD should report JAX GPU backend and same-as-real N_prime.",
    ),
    MethodSpec(
        method="rap_softmax",
        root="rap",
        gpu_policy="required",
        expected_env="tddpm",
        pattern_template="{dataset}/rho1p0/seed{seed}_T30K30M*I1000/run_metadata.json",
        expected_label="RAP softmax",
        note="RAP should report torch CUDA for the paper stress runs.",
    ),
    MethodSpec(
        method="private_pgm_aim",
        root="private_pgm_aim",
        gpu_policy="cpu_native",
        expected_env="baseline_mbi",
        pattern_template="{dataset}/rho1p0/seed{seed}/run_metadata.json",
        expected_label="Private-PGM AIM",
        note="Private-PGM AIM is MBI/private-pgm CPU-native in this environment.",
    ),
    MethodSpec(
        method="private_pgm_mst",
        root="private_pgm_mst",
        gpu_policy="cpu_native",
        expected_env="baseline_mbi",
        pattern_template="{dataset}/rho1p0/seed{seed}/run_metadata.json",
        expected_label="Private-PGM MST",
        note="Private-PGM MST is MBI/private-pgm CPU-native in this environment.",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _lower_list(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(value).lower() for value in values]
    if values is None:
        return []
    return [str(values).lower()]


def _truthy_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:
        return False


def _gpu_evidence_from_metadata(metadata: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    notes = metadata.get("notes") or {}
    torch_device = str(metadata.get("torch_device", ""))
    if metadata.get("torch_cuda_available") is True and torch_device.startswith("cuda"):
        evidence.append(f"torch_device={torch_device}")

    for source_name, source in (("metadata", metadata), ("notes", notes)):
        backend = str(source.get("jax_backend") or source.get("jax_default_backend") or "").lower()
        if backend == "gpu":
            evidence.append(f"{source_name}.jax_backend=gpu")
        devices = _lower_list(source.get("jax_devices") or source.get("gpu_devices"))
        cuda_devices = [device for device in devices if device.startswith(("cuda", "gpu"))]
        if cuda_devices:
            evidence.append(f"{source_name}.devices={','.join(cuda_devices)}")
        if _truthy_int(source.get("gpu_device_count")):
            evidence.append(f"{source_name}.gpu_device_count={source.get('gpu_device_count')}")
        if "gpu" in str(source.get("score_backend", "")).lower():
            evidence.append(f"{source_name}.score_backend={source.get('score_backend')}")
        if str(source.get("device", "")).lower() == "gpu":
            evidence.append(f"{source_name}.device=gpu")

    probe = metadata.get("jax_device_probe") or {}
    platforms = _lower_list(probe.get("jax_device_platforms"))
    devices = _lower_list(probe.get("jax_devices"))
    if "gpu" in platforms:
        evidence.append("jax_device_probe.platforms=gpu")
    cuda_probe = [device for device in devices if device.startswith("cuda")]
    if cuda_probe:
        evidence.append(f"jax_device_probe.devices={','.join(cuda_probe)}")

    if _truthy_int(notes.get("cupy_device_count")):
        evidence.append(f"notes.cupy_device_count={notes.get('cupy_device_count')}")
    return evidence


def _gpu_evidence_from_sidecars(run_dir: Path) -> list[str]:
    evidence: list[str] = []
    runtime = _read_json(run_dir / "runtime.json")
    metrics = _read_json(run_dir / "metrics_final.json")

    runtime_devices = _lower_list(runtime.get("gpu_devices"))
    cuda_devices = [device for device in runtime_devices if device.startswith(("cuda", "gpu"))]
    if cuda_devices:
        evidence.append(f"runtime.gpu_devices={','.join(cuda_devices)}")
    if "gpu" in str(runtime.get("score_backend", "")).lower():
        evidence.append(f"runtime.score_backend={runtime.get('score_backend')}")

    if _truthy_int(metrics.get("gpu_device_count")):
        evidence.append(f"metrics.gpu_device_count={metrics.get('gpu_device_count')}")
    if "gpu" in str(metrics.get("score_backend", "")).lower():
        evidence.append(f"metrics.score_backend={metrics.get('score_backend')}")

    config_path = run_dir / "config_resolved.yaml"
    if config_path.exists():
        text = config_path.read_text()
        if "device: gpu" in text:
            evidence.append("config.device=gpu")
        if "score_backend: dense_gpu" in text or "score_backend: sparse_delta_gpu" in text:
            evidence.append("config.score_backend=gpu")
    return evidence


def _isclose(value: Any, expected: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    try:
        return abs(float(value) - float(expected)) <= max(abs_tol, rel_tol * abs(float(expected)))
    except Exception:
        return False


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _command_has(command: str, flag: str, value: str) -> bool:
    tokens = command.split()
    try:
        return tokens[tokens.index(flag) + 1] == value
    except Exception:
        return False


def _protocol_evidence_from_metadata(spec: MethodSpec, dataset: str, metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    failures: list[str] = []
    notes = metadata.get("notes") or {}
    command = str(metadata.get("command") or "")

    def require(condition: bool, ok_label: str, failure_label: str) -> None:
        if condition:
            evidence.append(ok_label)
        else:
            failures.append(failure_label)

    require(metadata.get("conda_env") == spec.expected_env, f"conda_env={spec.expected_env}", "wrong_conda_env")
    require(str(metadata.get("method")) == spec.method, f"method={spec.method}", "wrong_method_label")
    require(_isclose(metadata.get("rho_total"), 1.0), "rho_total=1.0", "wrong_rho_total")
    require(metadata.get("n_real") == metadata.get("n_synthetic"), "n_synthetic=n_real", "synthetic_size_not_real")

    if spec.method == "sage":
        require(_as_int(notes.get("max_iters")) == 5000, "sage_max_iters=5000", "sage_max_iters_not_5000")
        require(_command_has(command, "--max-iters", "5000"), "command.max_iters=5000", "command_missing_sage_max_iters")
    elif spec.method == "private_gsd_gpu_1m_fulln_audit":
        genetic_operators = notes.get("genetic_operators")
        if isinstance(genetic_operators, str):
            genetic_operators = [item.strip() for item in genetic_operators.split(",") if item.strip()]
        require(notes.get("n_prime") == metadata.get("n_real"), "gsd_n_prime=n_real", "gsd_n_prime_not_real")
        require(_as_int(notes.get("num_generations")) == 1_000_000, "gsd_num_generations=1000000", "gsd_num_generations_not_1000000")
        require(
            _as_int(notes.get("stop_early_min_generation")) == 1_000_000,
            "gsd_stop_early_min_generation=1000000",
            "gsd_stop_early_min_generation_not_1000000",
        )
        require(_as_int(notes.get("tree_query_depth")) == 2, "gsd_tree_query_depth=2", "gsd_tree_query_depth_not_2")
        require(_isclose(notes.get("early_stop_threshold"), 0.01), "gsd_early_stop_threshold=0.01", "gsd_early_stop_threshold_not_0.01")
        require(
            list(genetic_operators or []) == ["mutate", "swap", "cross"],
            "gsd_genetic_operators=mutate,swap,cross",
            "gsd_genetic_operators_not_full",
        )
    elif spec.method == "rap_softmax":
        expected_marginals = 364 if dataset.startswith("br2000") else 445
        require(_as_int(metadata.get("model_rows")) == 1000, "rap_model_rows=1000", "rap_model_rows_not_1000")
        require(_as_int(notes.get("T")) == 30, "rap_T=30", "rap_T_not_30")
        require(_as_int(notes.get("K")) == 30, "rap_K=30", "rap_K_not_30")
        require(_as_int(notes.get("num_marginals")) == expected_marginals, f"rap_num_marginals={expected_marginals}", "rap_num_marginals_wrong")
        require(_as_int(notes.get("max_iters")) == 1000, "rap_max_iters=1000", "rap_max_iters_not_1000")
        require(str(notes.get("decode_mode")) == "sample", "rap_decode_mode=sample", "rap_decode_mode_not_sample")
    elif spec.method == "private_pgm_aim":
        require(_as_int(notes.get("rounds")) == 220, "aim_rounds=220", "aim_rounds_not_220")
        require(_as_int(notes.get("max_iters")) == 100, "aim_max_iters=100", "aim_max_iters_not_100")
        require(_isclose(notes.get("max_model_size"), 80.0), "aim_max_model_size=80.0", "aim_max_model_size_not_80")
    elif spec.method == "private_pgm_mst":
        require(_command_has(command, "--method", "mst"), "mst_method=mst", "mst_method_not_mst")

    if failures:
        evidence.extend(f"FAIL:{failure}" for failure in failures)
    return not failures, evidence


def _resolve_metadata_paths(runs_root: Path, spec: MethodSpec, dataset: str, seed: int) -> list[Path]:
    pattern = spec.pattern_template.format(dataset=dataset, seed=seed)
    return sorted((runs_root / spec.root).glob(pattern))


def _row_for_run(runs_root: Path, spec: MethodSpec, dataset: str, seed: int) -> dict[str, Any]:
    paths = _resolve_metadata_paths(runs_root, spec, dataset, seed)
    if not paths:
        return {
            "method": spec.method,
            "label": spec.expected_label,
            "dataset": dataset,
            "seed": seed,
            "status": "missing",
            "gpu_policy": spec.gpu_policy,
            "gpu_ok": "false" if spec.gpu_policy == "required" else "",
            "protocol_ok": "false",
            "runtime_seconds": "",
            "run_dir": "",
            "evidence": "",
            "protocol_evidence": "",
            "note": spec.note,
        }
    if len(paths) > 1:
        selected = max(paths, key=lambda path: path.stat().st_mtime)
    else:
        selected = paths[0]
    metadata = _read_json(selected)
    run_dir = selected.parent
    evidence = _gpu_evidence_from_metadata(metadata) + _gpu_evidence_from_sidecars(run_dir)
    protocol_ok, protocol_evidence = _protocol_evidence_from_metadata(spec, dataset, metadata)
    status = str(metadata.get("status") or "unknown")
    gpu_ok = bool(evidence) if spec.gpu_policy == "required" else ""
    if spec.gpu_policy == "cpu_native":
        gpu_ok = "not_required"
    return {
        "method": spec.method,
        "label": spec.expected_label,
        "dataset": dataset,
        "seed": seed,
        "status": status,
        "gpu_policy": spec.gpu_policy,
        "gpu_ok": gpu_ok,
        "protocol_ok": protocol_ok,
        "runtime_seconds": metadata.get("runtime_seconds", ""),
        "run_dir": str(run_dir),
        "evidence": "; ".join(dict.fromkeys(evidence)),
        "protocol_evidence": "; ".join(dict.fromkeys(protocol_evidence)),
        "note": spec.note,
    }


def audit(runs_root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in METHODS:
        for dataset in datasets:
            for seed in seeds:
                rows.append(_row_for_run(runs_root, spec, dataset, seed))
    return rows


def _is_failure(row: dict[str, Any]) -> bool:
    if row["status"] != "completed":
        return True
    if row["gpu_policy"] == "required" and row["gpu_ok"] is not True:
        return True
    if row["protocol_ok"] is not True:
        return True
    return False


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    lines = [
        "# Baseline GPU Provenance Audit",
        "",
        "This audit records whether each paper-facing run has direct GPU evidence or is explicitly CPU-native.",
        "It is not an accuracy comparison; it is a provenance guard for experiment execution.",
        "",
        "| Method | Policy | Completed | GPU/Protocol OK | Missing/Failed | Evidence examples |",
        "|---|---|---:|---:|---:|---|",
    ]
    for method, method_rows in grouped.items():
        completed = sum(1 for row in method_rows if row["status"] == "completed")
        gpu_ok = sum(1 for row in method_rows if row["gpu_ok"] is True or row["gpu_ok"] == "not_required")
        protocol_ok = sum(1 for row in method_rows if row["protocol_ok"] is True)
        failed = sum(1 for row in method_rows if _is_failure(row))
        examples = [row["evidence"] for row in method_rows if row["evidence"]]
        example = examples[0] if examples else method_rows[0]["note"]
        protocol_examples = [row["protocol_evidence"] for row in method_rows if row["protocol_evidence"]]
        if protocol_examples:
            example = f"{example}; {protocol_examples[0]}"
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(method_rows[0]["gpu_policy"]),
                    str(completed),
                    f"{gpu_ok}/{protocol_ok}",
                    str(failed),
                    example.replace("|", "/"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Policy Notes",
        "",
    ]
    for spec in METHODS:
        lines.append(f"- **{spec.expected_label}**: `{spec.gpu_policy}`. {spec.note}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GPU provenance for paper-facing baseline runs.")
    parser.add_argument("--runs-root", type=Path, default=external_runs())
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=external_results() / "baseline_gpu_provenance_20260707.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=external_results() / "baseline_gpu_provenance_20260707.md",
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = tuple(item for item in str(args.datasets).split(",") if item)
    seeds = tuple(int(item) for item in str(args.seeds).split(",") if item)
    rows = audit(args.runs_root, datasets, seeds)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)

    failures = [row for row in rows if _is_failure(row)]
    print(f"wrote {len(rows)} provenance rows to {args.output_csv}")
    print(f"wrote markdown to {args.output_md}")
    if failures:
        print("GPU provenance audit failed:")
        for row in failures[:20]:
            print(
                f"- {row['method']} {row['dataset']} seed{row['seed']}: "
                f"status={row['status']} gpu_ok={row['gpu_ok']}"
            )
        if len(failures) > 20:
            print(f"- ... {len(failures) - 20} more")
        return 1
    print("GPU provenance audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
