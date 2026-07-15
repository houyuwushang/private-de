from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from qdte.dataio import read_json, write_json
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "materialize_teacher_target.py"
    spec = importlib.util.spec_from_file_location("materialize_teacher_target", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    schema = TableSchema(
        columns=[ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"])]
    )
    schema.save_json(artifact / "schema.json")
    schema.save_json(input_dir / "schema.json")
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], name="a=0", group="a", family="oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], name="a=1", group="a", family="oneway")
    qcat = builder.build()
    qcat.save_json(artifact / "queries.json")
    qcat.save_json(input_dir / "queries_full.json")
    write_json(
        {
            "target_projected": [0.0, 0.0],
            "target_noisy": [0.0, 0.0],
            "variances": [2.0, 4.0],
            "groups": [{"name": "a", "query_indices": [0, 1], "is_partition": True}],
        },
        artifact / "measurements.json",
    )
    teacher_path = tmp_path / "teacher.npy"
    np.save(teacher_path, np.asarray([[0], [1], [1], [1]], dtype=np.int32))
    return artifact, input_dir, teacher_path


def test_materialize_teacher_target_uses_synthetic_answers_without_true_answers(tmp_path: Path) -> None:
    module = _load_module()
    artifact, input_dir, teacher_path = _write_tiny_artifact(tmp_path)
    output_dir = tmp_path / "teacher_target"

    metadata = module.run(
        argparse.Namespace(
            measurement=artifact,
            teacher_synthetic=teacher_path,
            output_dir=output_dir,
            input_dir=input_dir,
            queries=None,
            schema=None,
            batch_size=128,
            variant_name="teacher",
        )
    )

    measurement = read_json(output_dir / "measurements.json")
    assert measurement["target_projected"] == [1.0, 3.0]
    assert measurement["target_noisy"] == [1.0, 3.0]
    assert measurement["variances"] == [2.0, 4.0]
    assert measurement["projection_diagnostics"]["teacher_target"]["teacher_target_measured_self_loss"] == 0.0
    assert metadata["offline_true_answers_used"] is False
    assert (output_dir / "queries.json").exists()
    assert (output_dir / "schema.json").exists()
    assert (output_dir / "teacher_target_materialization_metadata.json").exists()


def test_materialize_teacher_target_validates_schema_width(tmp_path: Path) -> None:
    module = _load_module()
    artifact, input_dir, teacher_path = _write_tiny_artifact(tmp_path)
    np.save(teacher_path, np.asarray([[0, 1]], dtype=np.int32))

    try:
        module.run(
            argparse.Namespace(
                measurement=artifact,
                teacher_synthetic=teacher_path,
                output_dir=tmp_path / "bad",
                input_dir=input_dir,
                queries=None,
                schema=None,
                batch_size=128,
                variant_name="teacher",
            )
        )
    except ValueError as exc:
        assert "columns" in str(exc)
    else:
        raise AssertionError("expected schema-width validation failure")


def test_materialize_teacher_target_can_mix_two_synthetic_tables(tmp_path: Path) -> None:
    module = _load_module()
    artifact, input_dir, teacher_path = _write_tiny_artifact(tmp_path)
    secondary_path = tmp_path / "secondary.npy"
    np.save(secondary_path, np.asarray([[0], [0], [0], [0]], dtype=np.int32))
    output_dir = tmp_path / "mixture_target"

    metadata = module.run(
        argparse.Namespace(
            measurement=artifact,
            teacher_synthetic=teacher_path,
            secondary_synthetic=secondary_path,
            secondary_weight=0.5,
            output_dir=output_dir,
            input_dir=input_dir,
            queries=None,
            schema=None,
            batch_size=128,
            variant_name="mixture",
        )
    )

    measurement = read_json(output_dir / "measurements.json")
    assert measurement["target_projected"] == [2.5, 1.5]
    assert measurement["projection_diagnostics"]["teacher_target"]["method"] == (
        "synthetic_teacher_convex_mixture"
    )
    assert metadata["secondary_weight"] == 0.5


def test_anchor_balanced_lattice_rounding_preserves_partition_mass_and_anchor_ties(
    tmp_path: Path,
) -> None:
    module = _load_module()
    artifact, input_dir, teacher_path = _write_tiny_artifact(tmp_path)
    secondary_path = tmp_path / "secondary.npy"
    np.save(secondary_path, np.asarray([[0], [0], [0], [0]], dtype=np.int32))
    output_dir = tmp_path / "rounded_mixture_target"

    metadata = module.run(
        argparse.Namespace(
            measurement=artifact,
            teacher_synthetic=teacher_path,
            secondary_synthetic=secondary_path,
            secondary_weight=0.5,
            lattice_rounding="anchor_balanced",
            output_dir=output_dir,
            input_dir=input_dir,
            queries=None,
            schema=None,
            batch_size=128,
            variant_name="rounded_mixture",
        )
    )

    measurement = read_json(output_dir / "measurements.json")
    assert measurement["target_projected"] == [2.0, 2.0]
    rounding = measurement["projection_diagnostics"]["teacher_target"]["lattice_rounding"]
    assert rounding["mode"] == "anchor_balanced"
    assert rounding["num_fractional_queries_before"] == 2
    assert rounding["partition_mass_violations"] == 0
    assert sum(measurement["target_projected"]) == 4.0
    assert metadata["lattice_rounding"]["mode"] == "anchor_balanced"
