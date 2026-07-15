from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def released_measurement_target(
    measurement_dir: Path,
    num_queries: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    measurement_path = measurement_dir / "measurements.json"
    if not measurement_path.is_file():
        raise FileNotFoundError(f"measurement-dir has no measurements.json: {measurement_dir}")
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    if str(measurement.get("mode", "")).lower() != "dp":
        raise ValueError("--measurement-dir requires a released DP measurement artifact")
    target_counts = np.asarray(measurement["target_projected"], dtype=np.float64)
    variances = np.asarray(measurement["variances"], dtype=np.float64)
    if target_counts.shape != (num_queries,) or variances.shape != (num_queries,):
        raise ValueError("released target/variance vectors do not match queries.json")
    if not np.all(np.isfinite(target_counts)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("released target and variances must be finite with positive variance")
    num_rows = int(measurement.get("num_rows") or 0)
    if num_rows <= 0:
        raise ValueError("released DP measurement artifact must record a positive public num_rows")
    return target_counts, variances, num_rows, measurement


def normalized_precision_scale(variances: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    values = np.asarray(variances, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("variances must be a finite positive vector")
    inv_variance = 1.0 / values
    normalizer = float(np.mean(inv_variance))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("released precision normalization is invalid")
    return inv_variance, normalizer, np.sqrt(inv_variance / normalizer)
