from __future__ import annotations

import math

import numpy as np
import orjson

from qdte.eval.external import evaluate_external_synthetic
from qdte.queries.types import OP_EQ, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def test_external_evaluator_query_and_tvd_metrics(tmp_path) -> None:
    schema = TableSchema(
        columns=[
            ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["0", "1"]),
            ColumnSchema(name="b", kind="categorical", cardinality=2, categories=["0", "1"]),
        ]
    )
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], name="a=0", group="a", family="oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], name="a=1", group="a", family="oneway")
    builder.add([(1, OP_EQ, 0, 0, 0)], name="b=0", group="b", family="oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], name="b=1", group="b", family="oneway")
    qcat = builder.build()

    real = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    syn = np.asarray([[0, 0], [0, 0], [1, 0], [1, 0]], dtype=np.int32)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    schema.save_json(input_dir / "schema.json")
    qcat.save_json(input_dir / "queries_full.json")
    (input_dir / "metadata.json").write_text('{"dataset":"toy"}')
    (input_dir / "workload_groups.json").write_bytes(
        orjson.dumps(
            [
                {
                    "name": "a",
                    "family": "oneway",
                    "query_indices": [0, 1],
                    "sensitivity_l2": 1.0,
                    "is_partition": True,
                },
                {
                    "name": "b",
                    "family": "oneway",
                    "query_indices": [2, 3],
                    "sensitivity_l2": 1.0,
                    "is_partition": True,
                },
            ]
        )
    )
    np.save(input_dir / "real_encoded.npy", real)
    synthetic_path = tmp_path / "synthetic_encoded.npy"
    np.save(synthetic_path, syn)

    metrics = evaluate_external_synthetic(input_dir, synthetic_path, batch_size=2)

    assert metrics["dataset"] == "toy"
    assert metrics["num_queries"] == 4
    assert metrics["num_vector_blocks"] == 2
    assert math.isclose(metrics["full_true_mae"], 0.25)
    assert math.isclose(metrics["full_true_rmse"], math.sqrt(0.125))
    assert math.isclose(metrics["full_true_max_error"], 0.5)
    assert math.isclose(metrics["full_true_avg_tvd"], 0.25)
    assert math.isclose(metrics["full_true_max_tvd"], 0.5)
    assert metrics["per_family"]["oneway"]["num_queries"] == 4
    assert metrics["per_family_tvd"]["oneway"]["num_blocks"] == 2

    cache_path = tmp_path / "true_answers_cache.npz"
    cached_metrics = evaluate_external_synthetic(
        input_dir,
        synthetic_path,
        batch_size=2,
        true_answers_cache_path=cache_path,
    )
    assert cache_path.exists()
    assert cached_metrics["true_answers_cache"]["status"] == "miss_written"
    assert math.isclose(cached_metrics["full_true_mae"], metrics["full_true_mae"])
    assert math.isclose(cached_metrics["full_true_avg_tvd"], metrics["full_true_avg_tvd"])

    hit_metrics = evaluate_external_synthetic(
        input_dir,
        synthetic_path,
        batch_size=2,
        true_answers_cache_path=cache_path,
    )
    assert hit_metrics["true_answers_cache"]["status"] == "hit"
    assert math.isclose(hit_metrics["full_true_rmse"], metrics["full_true_rmse"])
    assert math.isclose(hit_metrics["full_true_max_tvd"], metrics["full_true_max_tvd"])
