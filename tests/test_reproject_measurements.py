from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from qdte.measurement.measure import MeasurementGroup, Measurements
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def _load_reproject_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reproject_measurements.py"
    spec = importlib.util.spec_from_file_location("reproject_measurements", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifact(tmp_path: Path) -> tuple[Path, Path]:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    schema = TableSchema(
        columns=[ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"])]
    )
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], name="a=0", group="a", family="oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], name="a=1", group="a", family="oneway")
    qcat = builder.build()
    group = MeasurementGroup(
        query_indices=np.asarray([0, 1], dtype=np.int32),
        sensitivity_l2=1.0,
        rho=1.0,
        sigma=1.0,
        noise_std=1.0,
        name="a",
        family="oneway",
        is_partition=True,
    )
    measurements = Measurements(
        target_noisy=np.asarray([5.0, -1.0], dtype=np.float32),
        target_projected=np.asarray([4.0, 0.0], dtype=np.float32),
        variances=np.asarray([1.0, 1.0], dtype=np.float32),
        inv_variances=np.asarray([1.0, 1.0], dtype=np.float32),
        groups=[group],
        mode="dp",
        rho_total=1.0,
        rho_spent=1.0,
        epsilon_delta=1.0,
        delta=1.0e-9,
        projection_diagnostics={"consistency": {"enabled": False}},
    )
    schema.save_json(artifact_dir / "schema.json")
    qcat.save_json(artifact_dir / "queries.json")
    module = _load_reproject_module()
    module.write_json(measurements.to_public_dict(), artifact_dir / "measurements.json")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
privacy:
  min_variance: 1.0e-6
projection:
  project_partitions: true
  clip_nonpartition: false
  prefix_monotonicity: false
""".lstrip(),
        encoding="utf-8",
    )
    return artifact_dir, config_path


def test_reproject_measurements_applies_configured_projection_without_true_metrics(tmp_path: Path) -> None:
    module = _load_reproject_module()
    artifact_dir, config_path = _write_artifact(tmp_path)
    output_dir = tmp_path / "reprojected"

    metadata = module.run(
        Namespace(
            measurement=artifact_dir,
            output_dir=output_dir,
            config=config_path,
            override=[],
            queries=None,
            schema=None,
            input_dir=None,
            real_encoded=None,
            total=4,
            projection_mode="configured",
            variant_name="simplex",
            seed=0,
            batch_size=2,
            compute_target_metrics=False,
        )
    )

    out = module.read_json(output_dir / "measurements.json")
    projected = np.asarray(out["target_projected"], dtype=np.float64)
    assert metadata["variant"] == "simplex"
    assert np.all(projected >= -1.0e-6)
    assert np.isclose(float(projected.sum()), 4.0)
    assert out["target_noisy"] == [5.0, -1.0]
    assert not (output_dir / "projection_target_metrics_offline.json").exists()
    assert module.read_json(output_dir / "reproject_metadata.json")["offline_true_metrics_computed"] is False


def test_reproject_uses_saved_variance_for_unadjusted_adaptive_artifact(tmp_path: Path) -> None:
    module = _load_reproject_module()
    artifact_dir, config_path = _write_artifact(tmp_path)
    payload = module.read_json(artifact_dir / "measurements.json")
    payload["variances"] = [0.25, 0.25]
    payload["projection_diagnostics"]["adaptive_selection"] = {"selected_blocks": 1}
    module.write_json(payload, artifact_dir / "measurements.json")

    output_dir = tmp_path / "reprojected"
    metadata = module.run(
        Namespace(
            measurement=artifact_dir,
            output_dir=output_dir,
            config=config_path,
            override=[],
            queries=None,
            schema=None,
            input_dir=None,
            real_encoded=None,
            total=4,
            projection_mode="configured",
            variant_name="adaptive_variance",
            seed=0,
            batch_size=2,
            compute_target_metrics=False,
        )
    )

    assert metadata["raw_variance_source"] == "artifact_variances"
    assert np.isclose(metadata["raw_variance_mean"], 0.25)
    out = module.read_json(output_dir / "measurements.json")
    assert np.allclose(out["variances"], [0.25, 0.25])


def test_reproject_measurements_writes_offline_target_metrics_only_when_requested(tmp_path: Path) -> None:
    module = _load_reproject_module()
    artifact_dir, config_path = _write_artifact(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "workload_groups.json").write_text(
        '[{"name":"a","family":"oneway","query_indices":[0,1],"sensitivity_l2":1.0,"is_partition":true}]',
        encoding="utf-8",
    )
    np.save(input_dir / "real_encoded.npy", np.asarray([[0], [0], [1], [1]], dtype=np.int32))
    output_dir = tmp_path / "with_metrics"

    module.run(
        Namespace(
            measurement=artifact_dir,
            output_dir=output_dir,
            config=config_path,
            override=[],
            queries=artifact_dir / "queries.json",
            schema=artifact_dir / "schema.json",
            input_dir=input_dir,
            real_encoded=None,
            total=4,
            projection_mode="configured",
            variant_name="simplex_metrics",
            seed=0,
            batch_size=2,
            compute_target_metrics=True,
        )
    )

    metrics = module.read_json(output_dir / "projection_target_metrics_offline.json")
    measurements = module.read_json(output_dir / "measurements.json")
    assert metrics["offline_true_answers_used"] is True
    assert metrics["offline_only"] is True
    assert "target_to_true_mae" in metrics
    assert "target_to_true_avg_tvd" in metrics
    assert "true_answers" not in measurements
