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

    @classmethod
    def from_dict(cls, data: dict) -> "TableSchema":
        cols = [ColumnSchema(**c) for c in data["columns"]]
        return cls(columns=cols, label_column=data.get("label_column"))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2))

    @classmethod
    def load_json(cls, path: str | Path) -> "TableSchema":
        return cls.from_dict(orjson.loads(Path(path).read_bytes()))
