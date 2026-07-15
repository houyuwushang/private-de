from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.gsd_released_target_utils import (
    normalized_precision_scale as _normalized_precision_scale,
    released_measurement_target,
)

try:
    from scripts.run_official_gsd_on_qdte_workload import (
        QDTEQueryStatistics,
        QueryData,
        run,
    )
except ImportError:
    QDTEQueryStatistics = None
    QueryData = None
    run = None


def _released_measurement_target(measurement_dir: Path, qcat: SimpleNamespace):
    return released_measurement_target(measurement_dir, qcat.m)


def _toy_queries() -> QueryData:
    if QueryData is None:
        pytest.skip("requires the pinned gsd environment")
    return QueryData(
        {
            "m": 2,
            "max_terms": 1,
            "attrs": [[0], [0]],
            "ops": [[0], [0]],
            "values": [[0], [1]],
            "lows": [[0], [1]],
            "highs": [[0], [1]],
            "num_terms": [1, 1],
            "linear_attrs": [[-1], [-1]],
            "linear_weights": [[0.0], [0.0]],
            "linear_thresholds": [0.0, 0.0],
            "linear_num_terms": [0, 0],
            "names": ["x=0", "x=1"],
            "groups": ["oneway:0", "oneway:0"],
            "families": ["oneway", "oneway"],
        }
    )


def test_released_measurement_target_loads_only_public_vectors(tmp_path: Path) -> None:
    artifact = {
        "mode": "dp",
        "target_projected": [6.0, 4.0],
        "variances": [4.0, 1.0],
        "num_rows": 10,
    }
    (tmp_path / "measurements.json").write_text(json.dumps(artifact), encoding="utf-8")
    target, variances, rows, stored = _released_measurement_target(
        tmp_path, SimpleNamespace(m=2)
    )
    np.testing.assert_array_equal(target, [6.0, 4.0])
    np.testing.assert_array_equal(variances, [4.0, 1.0])
    assert rows == 10
    assert stored == artifact


def test_released_measurement_target_rejects_non_dp(tmp_path: Path) -> None:
    artifact = {
        "mode": "oracle",
        "target_projected": [6.0, 4.0],
        "variances": [1.0, 1.0],
        "num_rows": 10,
    }
    (tmp_path / "measurements.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="released DP"):
        _released_measurement_target(tmp_path, SimpleNamespace(m=2))


def test_scaled_official_fitness_is_qdte_objective_up_to_global_constant() -> None:
    variances = np.asarray([4.0, 1.0, 0.25])
    rate_error = np.asarray([0.1, -0.2, 0.3])
    n_rows = 100
    inv, normalizer, scale = _normalized_precision_scale(variances)
    official_fitness = float(np.sum((rate_error * scale) ** 2))
    qdte_loss = float(0.5 * n_rows * n_rows * np.sum(rate_error * rate_error * inv))
    assert qdte_loss == pytest.approx(0.5 * n_rows * n_rows * normalizer * official_fitness)


@pytest.mark.skipif(QDTEQueryStatistics is None, reason="requires the pinned gsd environment")
def test_statistics_adapter_scales_target_and_synthetic_identically() -> None:
    qcat = _toy_queries()
    stats = QDTEQueryStatistics(
        None,
        qcat,
        np.asarray([0.6, 0.4]),
        statistic_scale=np.asarray([0.5, 2.0]),
    )
    np.testing.assert_allclose(np.asarray(stats.target_rates), [0.3, 0.8])
    synthetic = np.asarray([[0], [0], [1], [1]], dtype=np.int32)
    np.testing.assert_allclose(np.asarray(stats.rate_fn(synthetic)), [0.25, 1.0])


@pytest.mark.skipif(run is None, reason="requires the pinned gsd environment")
def test_one_generation_released_target_smoke_does_not_need_real_table(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    measurement_dir = tmp_path / "measurement"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    measurement_dir.mkdir()
    schema = {
        "columns": [
            {
                "name": "x",
                "kind": "categorical",
                "cardinality": 2,
                "categories": ["0", "1"],
                "bin_edges": None,
                "representatives": ["0", "1"],
                "missing_token": "__MISSING__",
            }
        ]
    }
    queries = _toy_queries().to_dict() if hasattr(_toy_queries(), "to_dict") else {
        "m": 2,
        "max_terms": 1,
        "attrs": [[0], [0]],
        "ops": [[0], [0]],
        "values": [[0], [1]],
        "lows": [[0], [1]],
        "highs": [[0], [1]],
        "num_terms": [1, 1],
        "linear_attrs": [[-1], [-1]],
        "linear_weights": [[0.0], [0.0]],
        "linear_thresholds": [0.0, 0.0],
        "linear_num_terms": [0, 0],
        "names": ["x=0", "x=1"],
        "groups": ["oneway:0", "oneway:0"],
        "families": ["oneway", "oneway"],
    }
    (input_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (measurement_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (measurement_dir / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    (measurement_dir / "measurements.json").write_text(
        json.dumps(
            {
                "mode": "dp",
                "target_projected": [3.0, 1.0],
                "variances": [4.0, 1.0],
                "num_rows": 4,
            }
        ),
        encoding="utf-8",
    )
    initial_path = tmp_path / "initial.npy"
    np.save(initial_path, np.asarray([[0], [0], [1], [1]], dtype=np.int32))
    args = argparse.Namespace(
        output_dir=output_dir,
        input_dir=input_dir,
        query_dir=None,
        measurement_dir=measurement_dir,
        n_syn="same_as_real",
        seed=0,
        num_generations=1,
        stop_early_min_generation=1,
        early_stop_threshold=0.0,
        plateau_min_generation=0,
        plateau_checkpoints=0,
        plateau_relative_improvement=0.0,
        initial_synthetic=initial_path,
        genetic_operators="mutate",
        verbose=False,
        method_label="official-gsd-released-target-smoke",
        dataset="toy",
        external_input_dir=None,
    )
    metrics = run(args)
    assert metrics["privacy_mode"] == "dp_released_target"
    assert metrics["measurement_reused"] is True
    assert metrics["num_generations"] == 1
    assert (output_dir / "synthetic_encoded.npy").is_file()
    assert not (input_dir / "real_encoded.npy").exists()
