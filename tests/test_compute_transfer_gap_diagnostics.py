from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compute_transfer_gap_diagnostics.py"
    spec = importlib.util.spec_from_file_location("compute_transfer_gap_diagnostics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _binary_partition_catalogue():
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], name="a=0", group="a", family="oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], name="a=1", group="a", family="oneway")
    return builder.build()


def test_weighted_transfer_components_close_exactly() -> None:
    module = _load_module()

    components = module._weighted_components(
        target=np.asarray([2.0, 1.0]),
        synthetic=np.asarray([1.0, 3.0]),
        true_answers=np.asarray([0.0, 2.0]),
        inv_variance=np.asarray([1.0, 2.0]),
    )

    assert np.isclose(
        components["E"],
        components["T"] + components["F"] + components["C"],
    )
    assert abs(components["closure"]) < 1.0e-12


def test_fractional_oracle_projects_to_reduced_row_realizable_simplex() -> None:
    module = _load_module()
    qcat = _binary_partition_catalogue()
    schema = TableSchema(
        columns=[ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"])]
    )
    measurement = {
        "target_projected": np.asarray([5.0, -1.0], dtype=np.float64),
        "target_noisy": np.asarray([5.0, -1.0], dtype=np.float64),
        "inv_variances": np.asarray([1.0, 1.0], dtype=np.float64),
    }

    row = module._run_fractional_oracle(
        dataset="tiny",
        seed=0,
        target_variant="baseline",
        qcat=qcat,
        schema=schema,
        measurement=measurement,
        true_answers=np.asarray([2.0, 2.0], dtype=np.float64),
        n_total=4,
        source_name="target_projected",
        attrs=[0],
        max_domain_cells=10,
        query_limit=10,
        max_iterations=100,
    )

    assert row["status"] == "ok"
    assert row["solver_success"] is True
    assert row["num_oracle_queries"] == 2
    assert row["oracle_domain_cells"] == 2
    assert np.isclose(row["source_to_oracle_objective"], 1.0, atol=1.0e-7)
    assert row["oracle_to_true_norm2"] <= row["source_to_true_norm2"]
