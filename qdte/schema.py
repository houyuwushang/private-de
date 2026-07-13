from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import orjson


ColumnKind = Literal["categorical", "numerical_binned"]


@dataclass
class ColumnSchema:
    name: str
    kind: ColumnKind
    cardinality: int
    categories: list[str] | None = None
    bin_edges: list[float] | None = None
    representatives: list[str] | None = None
    missing_token: str = "__MISSING__"


@dataclass
class TableSchema:
    columns: list[ColumnSchema]
    label_column: str | None = None

    @property
    def d(self) -> int:
        return len(self.columns)

    @property
    def cardinalities(self) -> np.ndarray:
        return np.asarray([c.cardinality for c in self.columns], dtype=np.int32)

    @property
    def numerical_indices(self) -> list[int]:
        return [i for i, col in enumerate(self.columns) if col.kind == "numerical_binned"]

    @property
    def categorical_indices(self) -> list[int]:
        return [i for i, col in enumerate(self.columns) if col.kind == "categorical"]

    def column_index(self, name: str) -> int:
        for idx, col in enumerate(self.columns):
            if col.name == name:
                return idx
        raise KeyError(name)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not self.columns:
            raise ValueError("Table schema must contain at least one column")
        names = [str(column.name) for column in self.columns]
        if any(not name for name in names):
            raise ValueError("Table schema column names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("Table schema column names must be unique")
        for index, column in enumerate(self.columns):
            if column.kind not in {"categorical", "numerical_binned"}:
                raise ValueError(f"Schema column {index} has unsupported kind {column.kind!r}")
            if int(column.cardinality) <= 0:
                raise ValueError(f"Schema column {index} must have positive cardinality")
            for field_name, values in (
                ("categories", column.categories),
                ("representatives", column.representatives),
            ):
                if values is not None and len(values) != int(column.cardinality):
                    raise ValueError(
                        f"Schema column {index} {field_name} length must equal cardinality "
                        f"{int(column.cardinality)}"
                    )
            if column.bin_edges is not None:
                edges = np.asarray(column.bin_edges, dtype=np.float64)
                if edges.ndim != 1 or not np.all(np.isfinite(edges)):
                    raise ValueError(f"Schema column {index} bin_edges must be a finite 1D vector")
                if len(edges) > 1 and np.any(np.diff(edges) <= 0.0):
                    raise ValueError(f"Schema column {index} bin_edges must be strictly increasing")
        if self.label_column is not None and str(self.label_column) not in set(names):
            raise ValueError(f"Schema label column {self.label_column!r} is not present in the table")

    @classmethod
    def from_dict(cls, data: dict) -> "TableSchema":
        cols = [ColumnSchema(**c) for c in data["columns"]]
        schema = cls(columns=cols, label_column=data.get("label_column"))
        schema.validate()
        return schema

    def save_json(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2))

    @classmethod
    def load_json(cls, path: str | Path) -> "TableSchema":
        return cls.from_dict(orjson.loads(Path(path).read_bytes()))
