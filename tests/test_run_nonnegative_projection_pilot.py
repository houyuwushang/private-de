from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from qdte.dataio import read_json, write_json
from qdte.queries.eval_jax import answer_queries
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_nonnegative_projection_pilot.py"
    spec = importlib.util.spec_from_file_location("run_nonnegative_projection_pilot", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_artifacts(tmp_path: Path):
    builder = QueryBuilder(max_terms=2)
    for attr in range(3):
        for value in range(2):
            builder.add(
                [(attr, OP_EQ, value, value, value)],
                f"x{attr}={value}",
                f"oneway:{attr}",
                "oneway",
            )
    for left in range(3):
        for right in range(left + 1, 3):
            for left_value in range(2):
                for right_value in range(2):
                    builder.add(
                        [
                            (left, OP_EQ, left_value, left_value, left_value),
                            (right, OP_EQ, right_value, right_value, right_value),
                        ],
                        f"x{left}={left_value}&x{right}={right_value}",
                        f"twoway:{left}:{right}",
                        "twoway",
                    )
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name=f"x{attr}",
                kind="categorical",
                cardinality=2,
                categories=["0", "1"],
            )
            for attr in range(3)
        ]
    )
    real = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    truth = answer_queries(real, qcat, batch_size=64).astype(np.float64)
    noise = np.linspace(-8.0, 9.0, qcat.m)
    variances = np.linspace(0.5, 5.0, qcat.m)
    measurement_path = tmp_path / "measurements.json"
    queries_path = tmp_path / "queries.json"
    schema_path = tmp_path / "schema.json"
    real_path = tmp_path / "real_encoded.npy"
    write_json(
        {
            "mode": "dp",
            "rho_total": 1.0,
            "rho_spent": 1.0,
            "delta": 1.0e-9,
            "epsilon_delta": 10.1,
            "target_noisy": (truth + noise).tolist(),
            "target_projected": (truth + noise).tolist(),
            "variances": variances.tolist(),
            "groups": [],
        },
        measurement_path,
    )
    qcat.save_json(queries_path)
    schema.save_json(schema_path)
    np.save(real_path, real)
    return measurement_path, queries_path, schema_path, real_path


def test_fixed_artifact_pilot_qualifies_t4(tmp_path: Path) -> None:
    module = _load_module()
    measurement, queries, schema, real = _toy_artifacts(tmp_path)

    summary = module.run_pilot(
        measurement_path=measurement,
        queries_path=queries,
        schema_path=schema,
        real_encoded_path=real,
        output_dir=tmp_path / "pilot",
        total=12,
        num_lowest_cardinality_attrs=2,
        max_scope_order=2,
        max_constraints=1000,
        max_dense_constraint_cells=100000,
        solver_max_iterations=1000,
        certificate_max_iterations=1000,
        batch_size=64,
        require_qualified=True,
        allow_resource_gate_failure=False,
    )

    assert summary["status"] == "theorem_qualified"
    assert summary["public_subset"]["selected_attrs"] == [0, 1]
    assert summary["fingerprints"]["match"] is True
    assert summary["solver"]["nonnegative_certificate_passed"] is True
    assert summary["theorem"]["dominance_holds_within_tolerance"] is True
    assert summary["dp_boundary"]["true_metrics_used_for_selection_or_solver"] is False
    assert (tmp_path / "pilot" / "pilot_summary.json").is_file()


def test_public_subset_rule_is_deterministic() -> None:
    module = _load_module()
    builder = QueryBuilder(max_terms=1)
    for attr, cardinality in enumerate([4, 2, 3]):
        for value in range(cardinality):
            builder.add(
                [(attr, OP_EQ, value, value, value)],
                f"x{attr}={value}",
                f"oneway:{attr}",
                "oneway",
            )
    qcat = builder.build()

    keep, metadata = module.select_public_query_subset(
        qcat,
        np.asarray([4, 2, 3], dtype=np.int32),
        num_lowest_cardinality_attrs=2,
        max_scope_order=1,
    )

    assert metadata["selected_attrs"] == [1, 2]
    assert len(keep) == 5


def test_full_subset_materializes_public_projected_measurement(tmp_path: Path) -> None:
    module = _load_module()
    measurement, queries, schema, real = _toy_artifacts(tmp_path)

    summary = module.run_pilot(
        measurement_path=measurement,
        queries_path=queries,
        schema_path=schema,
        real_encoded_path=real,
        output_dir=tmp_path / "pilot",
        total=12,
        num_lowest_cardinality_attrs=3,
        max_scope_order=2,
        max_constraints=1000,
        max_dense_constraint_cells=100000,
        solver_max_iterations=1000,
        certificate_max_iterations=1000,
        batch_size=64,
        require_qualified=True,
        allow_resource_gate_failure=False,
    )

    projected_path = tmp_path / "pilot" / "projected_measurement" / "measurements.json"
    source = read_json(measurement)
    projected = read_json(projected_path)
    assert summary["projected_measurement"]["path"] == str(projected_path)
    assert projected["target_noisy"] == source["target_noisy"]
    assert projected["variances"] == source["variances"]
    assert projected["projection_diagnostics"]["consistency"][
        "feasible_projection_certificate_passed"
    ] is True
    fixed = projected["projection_diagnostics"]["fixed_artifact_reprojection"]
    assert fixed["uses_true_answers"] is False
    assert "theorem" not in projected
    assert (projected_path.parent / "queries.json").is_file()
    assert (projected_path.parent / "schema.json").is_file()


def test_pilot_requires_public_total_for_old_artifact(tmp_path: Path) -> None:
    module = _load_module()
    measurement, queries, schema, real = _toy_artifacts(tmp_path)

    with pytest.raises(ValueError, match="public --total"):
        module.run_pilot(
            measurement_path=measurement,
            queries_path=queries,
            schema_path=schema,
            real_encoded_path=real,
            output_dir=tmp_path / "pilot",
            total=None,
            num_lowest_cardinality_attrs=2,
            max_scope_order=2,
            max_constraints=1000,
            max_dense_constraint_cells=100000,
            solver_max_iterations=1000,
            certificate_max_iterations=1000,
            batch_size=64,
            require_qualified=True,
            allow_resource_gate_failure=False,
        )


def test_full_resource_gate_fails_without_loading_truth(tmp_path: Path) -> None:
    module = _load_module()
    measurement, queries, schema, real = _toy_artifacts(tmp_path)

    summary = module.run_pilot(
        measurement_path=measurement,
        queries_path=queries,
        schema_path=schema,
        real_encoded_path=real,
        output_dir=tmp_path / "pilot",
        total=12,
        num_lowest_cardinality_attrs=3,
        max_scope_order=2,
        max_constraints=1000,
        max_dense_constraint_cells=1,
        solver_max_iterations=1000,
        certificate_max_iterations=1000,
        batch_size=64,
        require_qualified=True,
        allow_resource_gate_failure=True,
    )

    assert summary["status"] == "resource_gate_failed"
    assert summary["dp_boundary"]["true_answers_loaded"] is False
    assert summary["resource"]["resource_gate_passed"] is False
