#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml, save_yaml, set_nested
from qdte.dataio import ensure_dir, read_json, save_npy, write_json
from qdte.evolution.candidates import CandidateBatch, compute_edit_cost
from qdte.evolution.engine import run_qdte
from qdte.evolution.scoring import compute_deltas, score_candidates
from qdte.evolution.transport import apply_edits, choose_transport_batch, select_top_nonconflicting
from qdte.measurement.measure import measurements_from_public_dict
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import QueryCatalogue
from qdte.schema import TableSchema


def _population_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("population", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("population must be a mapping")
    crossover = raw.get("crossover", {}) or {}
    if not isinstance(crossover, dict):
        raise ValueError("population.crossover must be a mapping")
    parallel = raw.get("parallel", {}) or {}
    if not isinstance(parallel, dict):
        raise ValueError("population.parallel must be a mapping")
    qdte_cfg = config.get("qdte", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    size = int(raw.get("size", 4))
    elite_count = int(raw.get("elite_count", 1))
    crossover_enabled = bool(crossover.get("enabled", raw.get("crossover_enabled", False)))
    crossover_mode = str(crossover.get("mode", raw.get("crossover_mode", "random_row"))).lower()
    default_children = max(0, size - elite_count)
    default_candidates = int(qdte_cfg.get("total_candidates_per_iter", 1024))
    return {
        "size": size,
        "elite_count": elite_count,
        "generations": int(raw.get("generations", 1)),
        "inner_iters": int(raw.get("inner_iters", qdte_cfg.get("max_iters", 100))),
        "seed_stride": int(raw.get("seed_stride", 1000)),
        "evaluate_individuals": bool(raw.get("evaluate_individuals", config.get("evaluation", {}).get("compute_true_query_error", True))),
        "crossover_enabled": crossover_enabled,
        "crossover_mode": crossover_mode,
        "crossover_children": int(
            crossover.get("children", raw.get("crossover_children", default_children if crossover_enabled else 0))
        ),
        "crossover_fraction": float(crossover.get("fraction", raw.get("crossover_fraction", 0.5))),
        "crossover_parent_pool": int(crossover.get("parent_pool", raw.get("crossover_parent_pool", 0))),
        "crossover_candidates": int(crossover.get("candidates", raw.get("crossover_candidates", default_candidates))),
        "crossover_max_edits": int(crossover.get("max_edits", raw.get("crossover_max_edits", 0))),
        "crossover_inner_iters": int(
            crossover.get(
                "inner_iters",
                raw.get("crossover_inner_iters", raw.get("inner_iters", qdte_cfg.get("max_iters", 100))),
            )
        ),
        "parallel_enabled": bool(parallel.get("enabled", raw.get("parallel_enabled", False))),
        "parallel_workers": int(parallel.get("workers", raw.get("parallel_workers", 0))),
        "parallel_workers_per_gpu": int(parallel.get("workers_per_gpu", raw.get("parallel_workers_per_gpu", 1))),
        "parallel_gpu_devices": parallel.get("gpu_devices", raw.get("parallel_gpu_devices", "auto")),
        "parallel_capture_output": bool(parallel.get("capture_output", raw.get("parallel_capture_output", True))),
        "runtime_xla_preallocate": bool(runtime_cfg.get("xla_preallocate", True)),
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, int, int]:
    return (float(row["final_measured_loss"]), int(row.get("generation", 0)), int(row["index"]))


def _metrics_row(
    *,
    kind: str,
    generation: int,
    slot: int,
    index: int,
    seed: int,
    output_dir: Path,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": kind,
        "generation": int(generation),
        "slot": int(slot),
        "index": int(index),
        "seed": int(seed),
        "output_dir": str(output_dir),
        "final_measured_loss": float(metrics["final_measured_loss"]),
        "final_unweighted_measured_loss": float(metrics["final_unweighted_measured_loss"]),
        "num_candidates_scored": int(metrics["num_candidates_scored"]),
        "num_accepted_edits": int(metrics["num_accepted_edits"]),
        "final_true_query_rmse": metrics.get("final_true_query_rmse"),
    }
    if extra:
        row.update(extra)
    return row


def _detect_gpu_devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        if devices:
            return devices
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            cwd=str(ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["0"]
    devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return devices or ["0"]


def _parse_gpu_devices(raw: Any) -> list[str]:
    if raw is None:
        return _detect_gpu_devices()
    if isinstance(raw, int):
        return [str(raw)]
    if isinstance(raw, (list, tuple)):
        devices = [str(item).strip() for item in raw if str(item).strip()]
        return devices or _detect_gpu_devices()
    text = str(raw).strip()
    if text == "" or text.lower() == "auto":
        return _detect_gpu_devices()
    return [part.strip() for part in text.split(",") if part.strip()]


def _parallel_worker_devices(pop: dict[str, Any]) -> list[str]:
    if not bool(pop["parallel_enabled"]):
        return []
    devices = _parse_gpu_devices(pop["parallel_gpu_devices"])
    if not devices:
        devices = ["0"]
    workers_per_gpu = int(pop["parallel_workers_per_gpu"])
    slots: list[str] = []
    for device in devices:
        slots.extend([device] * workers_per_gpu)
    requested_workers = int(pop["parallel_workers"])
    if requested_workers > 0:
        slots = [devices[idx % len(devices)] for idx in range(requested_workers)]
    return slots or [devices[0]]


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _run_qdte_subprocess(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    device: str,
    worker_slot: int,
    capture_output: bool,
    force_xla_preallocate_false: bool,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    worker_config = output_dir / "worker_config.yaml"
    worker_log = output_dir / "worker_stdout.log"
    save_yaml(cfg, worker_config)
    env = os.environ.copy()
    if str(device).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["QDTE_POPULATION_WORKER_SLOT"] = str(worker_slot)
    if force_xla_preallocate_false:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif bool(cfg.get("runtime", {}).get("xla_preallocate", True)):
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    else:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_qdte.py"), "--config", str(worker_config)]
    stdout_target: Any
    if capture_output:
        stdout_target = worker_log.open("w", encoding="utf-8")
    else:
        stdout_target = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=stdout_target,
            stderr=subprocess.STDOUT if capture_output else None,
        )
    finally:
        if capture_output and stdout_target is not None:
            stdout_target.close()
    if proc.returncode != 0:
        tail = _tail_text(worker_log)
        raise RuntimeError(
            f"QDTE worker failed with exit code {proc.returncode}: "
            f"output_dir={output_dir}, device={device}, worker_slot={worker_slot}\n{tail}"
        )
    metrics_path = output_dir / "metrics_final.json"
    if not metrics_path.exists():
        tail = _tail_text(worker_log)
        raise FileNotFoundError(f"QDTE worker did not write {metrics_path}\n{tail}")
    return read_json(metrics_path)


def _start_qdte_worker(
    *,
    device: str,
    worker_slot: int,
    pop: dict[str, Any],
    worker_log_dir: Path,
) -> subprocess.Popen[str]:
    ensure_dir(worker_log_dir)
    env = os.environ.copy()
    if str(device).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["QDTE_POPULATION_WORKER_SLOT"] = str(worker_slot)
    if int(pop["parallel_workers_per_gpu"]) > 1:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif bool(pop.get("runtime_xla_preallocate", False)):
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    else:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    stderr_path = worker_log_dir / f"worker_{worker_slot:03d}_stderr.log"
    stderr_file = stderr_path.open("w", encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "population_worker.py")]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
    )
    setattr(proc, "_qdte_stderr_file", stderr_file)
    return proc


def _stop_qdte_worker(proc: subprocess.Popen[str]) -> None:
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except BrokenPipeError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    stderr_file = getattr(proc, "_qdte_stderr_file", None)
    if stderr_file is not None:
        stderr_file.close()


def _run_qdte_worker_task(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    device: str,
    worker_slot: int,
    proc: subprocess.Popen[str],
) -> dict[str, Any]:
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError(f"Persistent QDTE worker has no pipes: worker_slot={worker_slot}, device={device}")
    ensure_dir(output_dir)
    worker_config = output_dir / "worker_config.yaml"
    worker_log = output_dir / "worker_stdout.log"
    save_yaml(cfg, worker_config)
    task = {
        "config_path": str(worker_config),
        "output_dir": str(output_dir),
        "log_path": str(worker_log),
    }
    try:
        proc.stdin.write(json.dumps(task) + "\n")
        proc.stdin.flush()
    except BrokenPipeError as exc:
        raise RuntimeError(f"Persistent QDTE worker pipe closed: worker_slot={worker_slot}, device={device}") from exc
    line = proc.stdout.readline()
    if line == "":
        tail = _tail_text(worker_log)
        raise RuntimeError(
            f"Persistent QDTE worker exited without response: "
            f"worker_slot={worker_slot}, device={device}, returncode={proc.poll()}\n{tail}"
        )
    response = json.loads(line)
    if not bool(response.get("ok", False)):
        tail = _tail_text(worker_log)
        raise RuntimeError(
            f"Persistent QDTE worker task failed: worker_slot={worker_slot}, device={device}\n"
            f"{response.get('error', '')}\n{tail}"
        )
    metrics_path = output_dir / "metrics_final.json"
    if not metrics_path.exists():
        tail = _tail_text(worker_log)
        raise FileNotFoundError(f"Persistent QDTE worker did not write {metrics_path}\n{tail}")
    return read_json(metrics_path)


def _run_population_job(
    *,
    config: dict[str, Any],
    pop: dict[str, Any],
    measurement_dir: Path,
    job: dict[str, Any],
    device: str | None = None,
    worker_slot: int = 0,
    worker_proc: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(str(job["output_dir"]))
    init_path = job.get("init_encoded_npy")
    if init_path is not None:
        init_path = Path(str(init_path))
    print(
        "Running population individual "
        f"generation={job['generation']}, slot={job['slot']}, kind={job['kind']}, "
        f"index={job['index']}, seed={job['seed']}, output={output_dir}"
        + (f", gpu={device}, worker_slot={worker_slot}" if device is not None else ""),
        flush=True,
    )
    cfg = _prepare_individual_config(
        config,
        measurement_dir=measurement_dir,
        output_dir=output_dir,
        seed=int(job["seed"]),
        inner_iters=int(job["inner_iters"]),
        evaluate=bool(pop["evaluate_individuals"]),
        init_encoded_npy=init_path,
    )
    if device is None:
        metrics = run_qdte(cfg)
    elif worker_proc is not None:
        metrics = _run_qdte_worker_task(
            cfg,
            output_dir=output_dir,
            device=device,
            worker_slot=worker_slot,
            proc=worker_proc,
        )
    else:
        force_no_prealloc = int(pop["parallel_workers_per_gpu"]) > 1
        metrics = _run_qdte_subprocess(
            cfg,
            output_dir=output_dir,
            device=device,
            worker_slot=worker_slot,
            capture_output=bool(pop["parallel_capture_output"]),
            force_xla_preallocate_false=force_no_prealloc,
        )
    row = _metrics_row(
        kind=str(job["kind"]),
        generation=int(job["generation"]),
        slot=int(job["slot"]),
        index=int(job["index"]),
        seed=int(job["seed"]),
        output_dir=output_dir,
        metrics=metrics,
        extra=job.get("extra", {}),
    )
    print(
        "Finished population individual "
        f"generation={row['generation']}, slot={row['slot']}, index={row['index']}, "
        f"loss={row['final_measured_loss']:.6f}, accepted={row['num_accepted_edits']}",
        flush=True,
    )
    return row


class _PopulationWorkerPool:
    def __init__(self, *, config: dict[str, Any], pop: dict[str, Any], measurement_dir: Path) -> None:
        self.config = config
        self.pop = pop
        self.measurement_dir = measurement_dir
        self.worker_devices = _parallel_worker_devices(pop)
        self.worker_log_dir = measurement_dir.parent / "population_worker_logs"
        self.procs: list[subprocess.Popen[str]] = []
        if self.worker_devices:
            print(
                "Starting persistent population GPU workers: "
                f"workers={len(self.worker_devices)}, gpu_slots={self.worker_devices}",
                flush=True,
            )
            self.procs = [
                _start_qdte_worker(
                    device=device,
                    worker_slot=slot,
                    pop=pop,
                    worker_log_dir=self.worker_log_dir,
                )
                for slot, device in enumerate(self.worker_devices)
            ]

    def close(self) -> None:
        for proc in self.procs:
            _stop_qdte_worker(proc)
        self.procs = []

    def run(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.worker_devices:
            return [
                _run_population_job(
                    config=self.config,
                    pop=self.pop,
                    measurement_dir=self.measurement_dir,
                    job=job,
                )
                for job in jobs
            ]

        print(
            "Running population jobs in parallel: "
            f"jobs={len(jobs)}, workers={len(self.worker_devices)}, gpu_slots={self.worker_devices}",
            flush=True,
        )
        task_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
        for idx, job in enumerate(jobs):
            task_queue.put((idx, job))
        results: list[dict[str, Any] | None] = [None] * len(jobs)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(worker_slot: int, device: str, proc: subprocess.Popen[str]) -> None:
            while True:
                try:
                    idx, job = task_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    results[idx] = _run_population_job(
                        config=self.config,
                        pop=self.pop,
                        measurement_dir=self.measurement_dir,
                        job=job,
                        device=device,
                        worker_slot=worker_slot,
                        worker_proc=proc,
                    )
                except BaseException as exc:  # pragma: no cover - exercised by integration failures.
                    with lock:
                        errors.append(exc)
                finally:
                    task_queue.task_done()

        threads = [
            threading.Thread(target=worker, args=(slot, device, proc), daemon=True)
            for slot, (device, proc) in enumerate(zip(self.worker_devices, self.procs, strict=True))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError(f"{len(errors)} population worker(s) failed") from errors[0]
        missing = [idx for idx, row in enumerate(results) if row is None]
        if missing:
            raise RuntimeError(f"Population worker results missing for job indices: {missing}")
        return [row for row in results if row is not None]


def _prepare_measurement_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.pop("measurement", None)
    set_nested(cfg, "run.output_dir", str(output_dir))
    set_nested(cfg, "qdte.max_iters", 0)
    set_nested(cfg, "population.enabled", False)
    set_nested(cfg, "evaluation.compute_true_query_error", False)
    set_nested(cfg, "evaluation.compute_heldout_query_error", False)
    set_nested(cfg, "evaluation.save_synthetic_csv", False)
    return cfg


def _prepare_individual_config(
    config: dict[str, Any],
    *,
    measurement_dir: Path,
    output_dir: Path,
    seed: int,
    inner_iters: int,
    evaluate: bool,
    init_encoded_npy: Path | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    set_nested(cfg, "run.output_dir", str(output_dir))
    set_nested(cfg, "run.seed", int(seed))
    set_nested(cfg, "measurement.reuse_from", str(measurement_dir))
    set_nested(cfg, "population.enabled", False)
    set_nested(cfg, "qdte.max_iters", int(inner_iters))
    if init_encoded_npy is not None:
        set_nested(cfg, "init.encoded_npy", str(init_encoded_npy))
    if not evaluate:
        set_nested(cfg, "evaluation.compute_true_query_error", False)
        set_nested(cfg, "evaluation.compute_heldout_query_error", False)
        set_nested(cfg, "evaluation.save_synthetic_csv", False)
    return cfg


def _load_synthetic(row: dict[str, Any]) -> np.ndarray:
    path = Path(str(row["output_dir"])) / "synthetic_encoded.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing parent synthetic table: {path}")
    return np.load(path).astype(np.int32)


def _row_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if parent_a.shape != parent_b.shape:
        raise ValueError(f"Parent shape mismatch: {parent_a.shape} versus {parent_b.shape}")
    n = int(parent_a.shape[0])
    if n == 0:
        return parent_a.copy(), 0
    mask = rng.random(n) < float(fraction)
    if n > 1 and not np.any(mask):
        mask[int(rng.integers(0, n))] = True
    if n > 1 and np.all(mask):
        mask[int(rng.integers(0, n))] = False
    child = parent_a.copy()
    child[mask] = parent_b[mask]
    return child, int(np.sum(mask))


def _objective_inv_variance(measurements: Any, mode: str) -> np.ndarray:
    normalized = str(mode).lower()
    if normalized == "unweighted":
        return np.ones_like(measurements.inv_variances, dtype=np.float32)
    if normalized == "variance":
        return measurements.inv_variances.astype(np.float32)
    raise ValueError("qdte.objective_weighting must be one of: unweighted, variance")


def _context_aware_crossover(
    *,
    recipient: np.ndarray,
    donor: np.ndarray,
    qcat: QueryCatalogue,
    schema: TableSchema,
    target: np.ndarray,
    inv_variance: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    candidate_count: int,
    max_edits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if recipient.shape != donor.shape:
        raise ValueError(f"Parent shape mismatch: {recipient.shape} versus {donor.shape}")
    if candidate_count <= 0 or max_edits <= 0:
        return recipient.copy(), {
            "context_aware_candidates": 0,
            "context_aware_positive_candidates": 0,
            "context_aware_accepted_edits": 0,
            "context_aware_batch_advantage": 0.0,
        }
    n = int(recipient.shape[0])
    row_ids = rng.integers(0, n, size=int(candidate_count), dtype=np.int32)
    donor_ids = rng.integers(0, n, size=int(candidate_count), dtype=np.int32)
    old_rows = recipient[row_ids].copy()
    new_rows = donor[donor_ids].copy()
    changed = np.any(old_rows != new_rows, axis=1)
    if not np.any(changed):
        return recipient.copy(), {
            "context_aware_candidates": 0,
            "context_aware_positive_candidates": 0,
            "context_aware_accepted_edits": 0,
            "context_aware_batch_advantage": 0.0,
        }
    row_ids = row_ids[changed]
    old_rows = old_rows[changed]
    new_rows = new_rows[changed]
    qdte_cfg = config.get("qdte", {})
    numerical_gamma = float(qdte_cfg.get("numerical_distance_gamma", 0.1))
    edit_cost = compute_edit_cost(old_rows, new_rows, schema, numerical_gamma)
    candidates = CandidateBatch(
        row_ids=row_ids,
        old_rows=old_rows,
        new_rows=new_rows,
        target_query_ids=np.full(len(row_ids), -1, dtype=np.int32),
        edit_cost=edit_cost.astype(np.float32),
        repair_type=np.full(len(row_ids), 40, dtype=np.int32),
        diagnostics={"context_aware_crossover_candidates": float(len(row_ids))},
    )
    answers = answer_queries(
        recipient,
        qcat,
        batch_size=int(config.get("runtime", {}).get("answer_batch_size", 8192)),
    )
    residual = (target - answers).astype(np.float32)
    advantages = score_candidates(
        candidates,
        residual,
        inv_variance,
        qcat,
        lambda_cost=float(qdte_cfg.get("lambda_cost", 0.01)),
        chunk_size=int(config.get("runtime", {}).get("scoring_chunk_size", 4096)),
        use_pmap=bool(config.get("runtime", {}).get("use_pmap", True)),
    )
    min_advantage = float(qdte_cfg.get("min_advantage", 1.0e-6))
    selected = select_top_nonconflicting(candidates, advantages, max_edits, min_advantage)
    deltas = compute_deltas(candidates.old_rows[selected], candidates.new_rows[selected], qcat)
    transport = choose_transport_batch(
        candidates,
        advantages,
        deltas,
        selected,
        residual,
        inv_variance,
        float(qdte_cfg.get("lambda_cost", 0.01)),
        prefix_strategy=str(qdte_cfg.get("transport_prefix_strategy", "best_advantage")),
    )
    child = recipient.copy()
    apply_edits(child, candidates, transport.accepted_indices)
    return child, {
        "context_aware_candidates": int(candidates.size),
        "context_aware_positive_candidates": int(np.sum(advantages > min_advantage)),
        "context_aware_selected_candidates": int(len(selected)),
        "context_aware_accepted_edits": int(len(transport.accepted_indices)),
        "context_aware_batch_advantage": float(transport.batch_advantage),
        "context_aware_mean_advantage": float(transport.mean_advantage),
    }


def _load_shared_context(config: dict[str, Any], measurement_dir: Path) -> tuple[TableSchema, QueryCatalogue, np.ndarray, np.ndarray]:
    schema = TableSchema.load_json(measurement_dir / "schema.json")
    qcat = QueryCatalogue.from_dict(read_json(measurement_dir / "queries.json"))
    measurements = measurements_from_public_dict(read_json(measurement_dir / "measurements.json"))
    inv_variance = _objective_inv_variance(
        measurements,
        str(config.get("qdte", {}).get("objective_weighting", "variance")),
    )
    return schema, qcat, measurements.target_projected.astype(np.float32), inv_variance.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a shared-target population wrapper around QDTE.")
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    pop = _population_config(config)
    if pop["size"] <= 0:
        raise ValueError("population.size must be positive")
    if pop["elite_count"] <= 0 or pop["elite_count"] > pop["size"]:
        raise ValueError("population.elite_count must be in [1, population.size]")
    if pop["generations"] <= 0:
        raise ValueError("population.generations must be positive")
    if pop["inner_iters"] < 0:
        raise ValueError("population.inner_iters must be non-negative")
    if pop["seed_stride"] <= 0:
        raise ValueError("population.seed_stride must be positive")
    if pop["crossover_children"] < 0:
        raise ValueError("population.crossover.children must be non-negative")
    if pop["crossover_mode"] not in {"random_row", "context_aware"}:
        raise ValueError("population.crossover.mode must be one of: random_row, context_aware")
    if pop["crossover_fraction"] < 0.0 or pop["crossover_fraction"] > 1.0:
        raise ValueError("population.crossover.fraction must be in [0, 1]")
    if pop["crossover_parent_pool"] < 0:
        raise ValueError("population.crossover.parent_pool must be non-negative")
    if pop["crossover_candidates"] < 0:
        raise ValueError("population.crossover.candidates must be non-negative")
    if pop["crossover_max_edits"] < 0:
        raise ValueError("population.crossover.max_edits must be non-negative")
    if pop["crossover_inner_iters"] < 0:
        raise ValueError("population.crossover.inner_iters must be non-negative")
    if pop["parallel_workers"] < 0:
        raise ValueError("population.parallel.workers must be non-negative")
    if pop["parallel_workers_per_gpu"] <= 0:
        raise ValueError("population.parallel.workers_per_gpu must be positive")

    root = ensure_dir(config.get("run", {}).get("output_dir", "outputs/qdte_population"))
    measurement_dir = root / "measurement"
    print(f"Population output dir: {root}", flush=True)
    print(f"Measurement artifact dir: {measurement_dir}", flush=True)

    measurement_metrics = run_qdte(_prepare_measurement_config(config, measurement_dir))
    worker_pool = _PopulationWorkerPool(config=config, pop=pop, measurement_dir=measurement_dir)

    try:
        if pop["generations"] > 1:
            _run_generational_population(config, pop, root, measurement_dir, measurement_metrics, worker_pool)
            return
        _run_one_shot_population(config, pop, root, measurement_dir, measurement_metrics, worker_pool)
    finally:
        worker_pool.close()


def _run_generational_population(
    config: dict[str, Any],
    pop: dict[str, Any],
    root: Path,
    measurement_dir: Path,
    measurement_metrics: dict[str, Any],
    worker_pool: _PopulationWorkerPool,
) -> None:
        base_seed = int(config.get("run", {}).get("seed", 0))
        rng = np.random.default_rng(base_seed + 7919)
        shared_context = None
        if pop["crossover_enabled"] and pop["crossover_mode"] == "context_aware":
            shared_context = _load_shared_context(config, measurement_dir)

        all_individuals: list[dict[str, Any]] = []
        generation_summaries: list[dict[str, Any]] = []
        previous_ranked: list[dict[str, Any]] = []
        next_index = 0

        for generation in range(pop["generations"]):
            generation_dir = ensure_dir(root / f"generation_{generation:04d}")
            specs: list[dict[str, Any]] = []
            if generation == 0:
                for slot in range(pop["size"]):
                    specs.append({"kind": "restart", "slot": int(slot), "init_encoded_npy": None, "extra": {}})
            else:
                parent_pool_size = pop["crossover_parent_pool"] or max(2, pop["elite_count"])
                parent_pool = previous_ranked[: min(len(previous_ranked), max(2, parent_pool_size))]
                clone_count = min(pop["elite_count"], pop["size"], len(previous_ranked))
                for slot in range(clone_count):
                    parent = previous_ranked[slot]
                    specs.append(
                        {
                            "kind": "elite_clone",
                            "slot": int(slot),
                            "init_encoded_npy": Path(str(parent["output_dir"])) / "synthetic_encoded.npy",
                            "extra": {
                                "parent_index": int(parent["index"]),
                                "parent_output_dir": str(parent["output_dir"]),
                            },
                        }
                    )

                slots_left = pop["size"] - len(specs)
                crossover_count = 0
                if pop["crossover_enabled"] and pop["crossover_children"] > 0 and slots_left > 0:
                    if len(parent_pool) < 2:
                        raise ValueError("population crossover requires at least two parent individuals")
                    crossover_count = min(pop["crossover_children"], slots_left)
                for local_child_idx in range(crossover_count):
                    slot = len(specs)
                    parent_ids = rng.choice(len(parent_pool), size=2, replace=False)
                    parent_a = parent_pool[int(parent_ids[0])]
                    parent_b = parent_pool[int(parent_ids[1])]
                    parent_a_table = _load_synthetic(parent_a)
                    parent_b_table = _load_synthetic(parent_b)
                    crossover_diag: dict[str, Any] = {}
                    if pop["crossover_mode"] == "context_aware":
                        if shared_context is None:
                            raise RuntimeError("Internal error: missing shared context for context-aware crossover")
                        schema, qcat, target, inv_variance = shared_context
                        inferred_max_edits = max(1, int(round(pop["crossover_fraction"] * parent_a_table.shape[0])))
                        max_edits = pop["crossover_max_edits"] or inferred_max_edits
                        child_table, crossover_diag = _context_aware_crossover(
                            recipient=parent_a_table,
                            donor=parent_b_table,
                            qcat=qcat,
                            schema=schema,
                            target=target,
                            inv_variance=inv_variance,
                            config=config,
                            rng=rng,
                            candidate_count=pop["crossover_candidates"],
                            max_edits=max_edits,
                        )
                        swapped_rows = int(crossover_diag["context_aware_accepted_edits"])
                    else:
                        child_table, swapped_rows = _row_crossover(
                            parent_a_table,
                            parent_b_table,
                            pop["crossover_fraction"],
                            rng,
                        )
                    init_path = generation_dir / f"crossover_initial_slot_{slot:03d}.npy"
                    save_npy(child_table, init_path)
                    specs.append(
                        {
                            "kind": "crossover",
                            "slot": int(slot),
                            "init_encoded_npy": init_path,
                            "extra": {
                                "parent_indices": [int(parent_a["index"]), int(parent_b["index"])],
                                "parent_output_dirs": [str(parent_a["output_dir"]), str(parent_b["output_dir"])],
                                "crossover_mode": str(pop["crossover_mode"]),
                                "swapped_rows": int(swapped_rows),
                                "crossover_fraction": float(pop["crossover_fraction"]),
                                **crossover_diag,
                            },
                        }
                    )

                while len(specs) < pop["size"]:
                    specs.append(
                        {
                            "kind": "restart",
                            "slot": int(len(specs)),
                            "init_encoded_npy": None,
                            "extra": {},
                        }
                    )

            print(
                f"Running generation {generation + 1}/{pop['generations']}: "
                f"individuals={len(specs)}, inner_iters={pop['inner_iters']}",
                flush=True,
            )
            jobs: list[dict[str, Any]] = []
            for spec in specs:
                index = next_index
                next_index += 1
                seed = base_seed + index * pop["seed_stride"]
                slot = int(spec["slot"])
                out = generation_dir / f"individual_{slot:03d}"
                jobs.append(
                    {
                        "kind": str(spec["kind"]),
                        "generation": int(generation),
                        "slot": slot,
                        "index": index,
                        "seed": seed,
                        "output_dir": out,
                        "inner_iters": int(pop["inner_iters"]),
                        "init_encoded_npy": spec["init_encoded_npy"],
                        "extra": spec["extra"],
                    }
                )
            generation_rows = worker_pool.run(jobs)

            all_individuals.extend(generation_rows)
            previous_ranked = sorted(generation_rows, key=_rank_key)
            global_best = min(all_individuals, key=_rank_key)
            generation_summaries.append(
                {
                    "generation": int(generation),
                    "best_index": int(previous_ranked[0]["index"]),
                    "best_measured_loss": float(previous_ranked[0]["final_measured_loss"]),
                    "global_best_index": int(global_best["index"]),
                    "global_best_measured_loss": float(global_best["final_measured_loss"]),
                    "num_individuals": int(len(generation_rows)),
                    "num_candidates_scored": int(sum(row["num_candidates_scored"] for row in generation_rows)),
                    "num_accepted_edits": int(sum(row["num_accepted_edits"] for row in generation_rows)),
                }
            )
            print(
                f"Generation {generation} best loss={previous_ranked[0]['final_measured_loss']:.6f}; "
                f"global best loss={global_best['final_measured_loss']:.6f}",
                flush=True,
            )

        ranked = sorted(all_individuals, key=_rank_key)
        elites = ranked[: pop["elite_count"]]
        summary_mode = "generational_random_restart_elite"
        if pop["crossover_enabled"]:
            summary_mode = f"generational_random_restart_elite_{pop['crossover_mode']}_crossover"
        summary = {
            "mode": summary_mode,
            "selection_metric": "final_measured_loss",
            "measurement_dir": str(measurement_dir),
            "measurement_metrics": {
                "privacy_mode": measurement_metrics.get("privacy_mode"),
                "rho_total": measurement_metrics.get("rho_total"),
                "rho_spent": measurement_metrics.get("rho_spent"),
                "epsilon_delta": measurement_metrics.get("epsilon_delta"),
                "num_queries": measurement_metrics.get("num_queries"),
                "objective_weighting": measurement_metrics.get("objective_weighting"),
            },
            "population": pop,
            "generation_summaries": generation_summaries,
            "individuals": all_individuals,
            "elites": elites,
            "best": elites[0] if elites else None,
        }
        write_json(summary, root / "population_summary.json")
        print(f"Best individual: {summary['best']}", flush=True)


def _run_one_shot_population(
    config: dict[str, Any],
    pop: dict[str, Any],
    root: Path,
    measurement_dir: Path,
    measurement_metrics: dict[str, Any],
    worker_pool: _PopulationWorkerPool,
) -> None:
    base_seed = int(config.get("run", {}).get("seed", 0))
    restart_jobs: list[dict[str, Any]] = []
    for idx in range(pop["size"]):
        seed = base_seed + idx * pop["seed_stride"]
        out = root / f"individual_{idx:03d}"
        restart_jobs.append(
            {
                "kind": "restart",
                "generation": 0,
                "slot": int(idx),
                "index": int(idx),
                "seed": seed,
                "output_dir": out,
                "inner_iters": int(pop["inner_iters"]),
                "init_encoded_npy": None,
                "extra": {},
            }
        )
    individuals = worker_pool.run(restart_jobs)

    ranked = sorted(individuals, key=_rank_key)
    if pop["crossover_enabled"] and pop["crossover_children"] > 0:
        if len(ranked) < 2:
            raise ValueError("population crossover requires at least two parent individuals")
        rng = np.random.default_rng(base_seed + 7919)
        parent_pool_size = pop["crossover_parent_pool"] or max(2, pop["elite_count"])
        parent_pool = ranked[: min(len(ranked), max(2, parent_pool_size))]
        shared_context = None
        if pop["crossover_mode"] == "context_aware":
            shared_context = _load_shared_context(config, measurement_dir)
        child_jobs: list[dict[str, Any]] = []
        for child_idx in range(pop["crossover_children"]):
            parent_ids = rng.choice(len(parent_pool), size=2, replace=False)
            parent_a = parent_pool[int(parent_ids[0])]
            parent_b = parent_pool[int(parent_ids[1])]
            child_dir = root / f"crossover_child_{child_idx:03d}"
            ensure_dir(child_dir)
            child_seed = base_seed + (pop["size"] + child_idx) * pop["seed_stride"]
            parent_a_table = _load_synthetic(parent_a)
            parent_b_table = _load_synthetic(parent_b)
            crossover_diag: dict[str, Any] = {}
            if pop["crossover_mode"] == "context_aware":
                if shared_context is None:
                    raise RuntimeError("Internal error: missing shared context for context-aware crossover")
                schema, qcat, target, inv_variance = shared_context
                inferred_max_edits = max(1, int(round(pop["crossover_fraction"] * parent_a_table.shape[0])))
                max_edits = pop["crossover_max_edits"] or inferred_max_edits
                child_table, crossover_diag = _context_aware_crossover(
                    recipient=parent_a_table,
                    donor=parent_b_table,
                    qcat=qcat,
                    schema=schema,
                    target=target,
                    inv_variance=inv_variance,
                    config=config,
                    rng=rng,
                    candidate_count=pop["crossover_candidates"],
                    max_edits=max_edits,
                )
                swapped_rows = int(crossover_diag["context_aware_accepted_edits"])
            else:
                child_table, swapped_rows = _row_crossover(
                    parent_a_table,
                    parent_b_table,
                    pop["crossover_fraction"],
                    rng,
                )
            init_path = child_dir / "crossover_initial.npy"
            save_npy(child_table, init_path)
            child_jobs.append(
                {
                    "kind": "crossover",
                    "generation": 0,
                    "slot": int(pop["size"] + child_idx),
                    "index": int(pop["size"] + child_idx),
                    "seed": int(child_seed),
                    "output_dir": child_dir,
                    "inner_iters": int(pop["crossover_inner_iters"]),
                    "init_encoded_npy": init_path,
                    "extra": {
                        "parent_indices": [int(parent_a["index"]), int(parent_b["index"])],
                        "parent_output_dirs": [str(parent_a["output_dir"]), str(parent_b["output_dir"])],
                        "crossover_mode": str(pop["crossover_mode"]),
                        "swapped_rows": int(swapped_rows),
                        "crossover_fraction": float(pop["crossover_fraction"]),
                        **crossover_diag,
                    },
                }
            )
        individuals.extend(
            worker_pool.run(child_jobs)
        )
        ranked = sorted(individuals, key=_rank_key)
    elites = ranked[: pop["elite_count"]]
    summary_mode = "random_restart_elite"
    if pop["crossover_enabled"]:
        summary_mode = f"random_restart_elite_{pop['crossover_mode']}_crossover"
    summary = {
        "mode": summary_mode,
        "selection_metric": "final_measured_loss",
        "measurement_dir": str(measurement_dir),
        "measurement_metrics": {
            "privacy_mode": measurement_metrics.get("privacy_mode"),
            "rho_total": measurement_metrics.get("rho_total"),
            "rho_spent": measurement_metrics.get("rho_spent"),
            "epsilon_delta": measurement_metrics.get("epsilon_delta"),
            "num_queries": measurement_metrics.get("num_queries"),
            "objective_weighting": measurement_metrics.get("objective_weighting"),
        },
        "population": pop,
        "individuals": individuals,
        "elites": elites,
        "best": elites[0] if elites else None,
    }
    write_json(summary, root / "population_summary.json")
    print(f"Best individual: {summary['best']}", flush=True)


if __name__ == "__main__":
    main()
