from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qdte.preprocess import decode_array, load_and_preprocess_csv
from qdte.schema import ColumnSchema, TableSchema


def _write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "cat": ["a", "b", "a", None],
            "num": [0.0, 1.0, 2.0, 3.0],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(path, index=False)
    return path


def test_preprocess_encodes_declared_public_schema_columns(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    result = load_and_preprocess_csv(
        {
            "run": {"input_csv": str(path)},
            "preprocess": {
                "categorical_columns": ["cat", "label"],
                "numerical_columns": ["num"],
                "numerical_bins": 2,
                "label_column": "label",
            },
        }
    )

    assert result.X.shape == (4, 3)
    assert result.schema.d == 3
    assert result.schema.label_column == "label"
    assert result.schema.columns[0].kind == "categorical"
    assert result.schema.columns[1].kind == "numerical_binned"
    assert np.all(result.X >= 0)
    for attr, cardinality in enumerate(result.schema.cardinalities.tolist()):
        assert np.all(result.X[:, attr] < cardinality)


def test_dp_release_mode_encodes_only_from_explicit_public_schema(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name="cat",
                kind="categorical",
                cardinality=3,
                categories=["a", "b", "__MISSING__"],
                representatives=["a", "b", "__MISSING__"],
            ),
            ColumnSchema(
                name="num",
                kind="numerical_binned",
                cardinality=2,
                bin_edges=[0.0, 2.0, 4.0],
                representatives=["low", "high"],
            ),
            ColumnSchema(
                name="label",
                kind="categorical",
                cardinality=2,
                categories=["0", "1"],
                representatives=["0", "1"],
            ),
        ],
        label_column="label",
    )
    schema_path = tmp_path / "schema.json"
    schema.save_json(schema_path)

    result = load_and_preprocess_csv(
        {
            "run": {"input_csv": str(path)},
            "privacy": {"mode": "dp", "dp_release_mode": True},
            "preprocess": {"public_schema_json": "sibling"},
        }
    )

    assert result.public_schema_path == schema_path.resolve()
    assert result.schema.to_dict() == schema.to_dict()
    assert result.X.tolist() == [[0, 0, 0], [1, 0, 1], [0, 1, 0], [2, 1, 1]]


def test_dp_release_mode_rejects_schema_inference(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)

    with pytest.raises(ValueError, match="public_schema_json"):
        load_and_preprocess_csv(
            {
                "run": {"input_csv": str(path)},
                "privacy": {"mode": "dp", "dp_release_mode": True},
                "preprocess": {},
            }
        )


def test_public_schema_rejects_unknown_category(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"cat": ["known", "unknown"]}).to_csv(path, index=False)
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name="cat",
                kind="categorical",
                cardinality=1,
                categories=["known"],
                representatives=["known"],
            )
        ]
    )
    schema_path = tmp_path / "schema.json"
    schema.save_json(schema_path)

    with pytest.raises(ValueError, match="absent from its public codebook"):
        load_and_preprocess_csv(
            {
                "run": {"input_csv": str(path)},
                "preprocess": {"public_schema_json": str(schema_path)},
            }
        )


@pytest.mark.parametrize(
    ("preprocess", "message"),
    [
        ({"categorical_columns": ["missing"]}, "missing from the CSV"),
        ({"categorical_columns": ["cat"], "numerical_columns": ["cat"]}, "both numerical and categorical"),
        ({"label_column": "missing"}, "label_column"),
        ({"numerical_bins": 0}, "numerical_bins"),
        ({"auto_numeric_min_unique": 0}, "auto_numeric_min_unique"),
    ],
)
def test_preprocess_rejects_ambiguous_or_invalid_schema_config(
    tmp_path: Path,
    preprocess: dict[str, object],
    message: str,
) -> None:
    path = _write_csv(tmp_path)

    with pytest.raises(ValueError, match=message):
        load_and_preprocess_csv({"run": {"input_csv": str(path)}, "preprocess": preprocess})


def test_schema_and_decoder_enforce_public_domain() -> None:
    schema = TableSchema(columns=[ColumnSchema(name="a", kind="categorical", cardinality=2)])
    schema.validate()
    with pytest.raises(ValueError, match="outside the public domain"):
        decode_array(np.asarray([[2]], dtype=np.int32), schema)

    invalid = TableSchema(
        columns=[ColumnSchema(name="a", kind="categorical", cardinality=2, categories=["only-one"])]
    )
    with pytest.raises(ValueError, match="length must equal cardinality"):
        invalid.validate()
