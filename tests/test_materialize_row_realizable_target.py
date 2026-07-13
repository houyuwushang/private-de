from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from qdte.dataio import read_json, write_json
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "materialize_row_realizable_target.py"
    spec = importlib.util.spec_from_file_location("materialize_row_realizable_target", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialize_reduced_target_without_true_answers(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    schema = TableSchema(
        columns=[ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"])]
    )
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], name="a=0", group="a", family="oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], name="a=1", group="a", family="oneway")
    qcat = builder.build()
    schema.save_json(artifact / "schema.json")
    qcat.save_json(artifact / "queries.json")
    write_json(
        {
            "mode": "dp",
            "rho_total": 1.0,
            "rho_spent": 1.0,
            "delta": 1.0e-9,
            "epsilon_delta": 1.0,
            "target_noisy": [5.0, -1.0],
            "target_projected": [5.0, -1.0],
            "variances": [1.0, 1.0],
            "groups": [
                {
                    "query_indices": [0, 1],
                    "sensitivity_l2": 1.0,
                    "rho": 1.0,
                    "sigma": 1.0,
                    "noise_std": 1.0,
                    "name": "a",
                    "family": "oneway",
                    "is_partition": True,
                }
            ],
            "projection_diagnostics": {},
        },
        artifact / "measurements.json",
    )
    output_dir = tmp_path / "rtp"

    metadata = module.run(
        Namespace(
            measurement=artifact,
            output_dir=output_dir,
            attrs="0",
            source="target_projected",
            input_dir=None,
            queries=None,
            schema=None,
            total=4,
            query_limit=10,
            max_domain_cells=10,
            max_iterations=100,
            variant_name="rtp_local",
            require_success=True,
        )
    )

    out = read_json(output_dir / "measurements.json")
    projected = np.asarray(out["target_projected"], dtype=np.float64)
    assert metadata["offline_true_answers_used"] is False
    assert np.allclose(projected, [4.0, 0.0], atol=1.0e-5)
    assert out["target_noisy"] == [5.0, -1.0]
    diag = out["projection_diagnostics"]["row_realizable_local"]
    assert diag["num_queries"] == 2
    assert diag["domain_cells"] == 2
    assert diag["solver_success"] is True
    assert diag["realizability_gap_objective"] > 0.0
    assert diag["source_objective_increase"] > 0.0
    assert "true_answers" not in out
