from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from qdte.schema import TableSchema


PUBLIC_LEGAL_ROW_DOMAIN_METHOD = "sealed_public_cartesian_row_domain_v1"


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


@dataclass(frozen=True)
class PublicLegalRowDomain:
    """Sealed public encoded-row domain used by the RHCG pricing oracle.

    Version 1 supports the full Cartesian product of explicitly declared
    categorical/bin IDs. Additional row constraints must fail closed until a
    matching exact MILP encoding is implemented.
    """

    attribute_names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    category_ids: tuple[tuple[str, ...], ...]
    missing_tokens: tuple[str, ...]
    public_n: int
    schema_hash: str
    declared_constraints: tuple[dict[str, Any], ...] = ()
    method: str = PUBLIC_LEGAL_ROW_DOMAIN_METHOD

    def __post_init__(self) -> None:
        if self.method != PUBLIC_LEGAL_ROW_DOMAIN_METHOD:
            raise ValueError(f"Unsupported public row-domain method {self.method!r}")
        if not self.attribute_names or len(self.attribute_names) != len(
            self.cardinalities
        ):
            raise ValueError("Public row domain requires schema-aligned attributes")
        if len(set(self.attribute_names)) != len(self.attribute_names):
            raise ValueError("Public row-domain attribute names must be unique")
        if any(int(value) < 2 for value in self.cardinalities):
            raise ValueError("RHCG requires public cardinalities of at least two")
        if len(self.category_ids) != self.dimension or len(self.missing_tokens) != self.dimension:
            raise ValueError("Public row-domain metadata must match its dimension")
        for cardinality, identifiers in zip(
            self.cardinalities, self.category_ids, strict=True
        ):
            if len(identifiers) != int(cardinality) or len(set(identifiers)) != len(
                identifiers
            ):
                raise ValueError("Public category IDs must be unique and complete")
        if int(self.public_n) <= 0:
            raise ValueError("RHCG requires a positive public row count")
        if self.declared_constraints:
            raise ValueError(
                "RHCG-CCMP-v1 does not yet implement additional public row constraints"
            )
        if len(self.schema_hash) != 64:
            raise ValueError("Public schema hash must be a SHA-256 hex digest")

    @classmethod
    def from_schema(
        cls,
        schema: TableSchema,
        *,
        public_n: int,
        declared_constraints: tuple[dict[str, Any], ...] = (),
    ) -> PublicLegalRowDomain:
        schema.validate()
        schema_payload = schema.to_dict()
        schema_hash = hashlib.sha256(_canonical_json(schema_payload)).hexdigest()
        category_ids: list[tuple[str, ...]] = []
        for column in schema.columns:
            declared = column.categories or column.representatives
            if declared is None:
                identifiers = tuple(str(value) for value in range(column.cardinality))
            else:
                identifiers = tuple(str(value) for value in declared)
            category_ids.append(identifiers)
        return cls(
            attribute_names=tuple(str(column.name) for column in schema.columns),
            cardinalities=tuple(int(column.cardinality) for column in schema.columns),
            category_ids=tuple(category_ids),
            missing_tokens=tuple(str(column.missing_token) for column in schema.columns),
            public_n=int(public_n),
            schema_hash=schema_hash,
            declared_constraints=tuple(declared_constraints),
        )

    @property
    def dimension(self) -> int:
        return len(self.cardinalities)

    @property
    def domain_size(self) -> int:
        return math.prod(self.cardinalities)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "attribute_order": list(self.attribute_names),
            "cardinalities": list(self.cardinalities),
            "category_id_order": [list(values) for values in self.category_ids],
            "missing_value_rules": list(self.missing_tokens),
            "public_n": int(self.public_n),
            "schema_sha256": self.schema_hash,
            "declared_constraints": list(self.declared_constraints),
            "domain_size": self.domain_size,
        }

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest)).hexdigest()

    def validate_rows(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.int32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("Rows must be a schema-aligned two-dimensional array")
        for attribute, cardinality in enumerate(self.cardinalities):
            if np.any(values[:, attribute] < 0) or np.any(
                values[:, attribute] >= cardinality
            ):
                raise ValueError("Rows contain a value outside the public domain")
        return values

    def enumerate_rows(self, *, maximum_rows: int | None = None) -> np.ndarray:
        if maximum_rows is not None and self.domain_size > int(maximum_rows):
            raise ValueError(
                f"Public row domain has {self.domain_size} rows, above cap {maximum_rows}"
            )
        rows = np.asarray(
            list(itertools.product(*(range(value) for value in self.cardinalities))),
            dtype=np.int32,
        )
        if rows.shape != (self.domain_size, self.dimension):
            raise RuntimeError("Public row enumeration produced an invalid shape")
        return rows

    def write_manifests(self, directory: str | Path, schema: TableSchema) -> dict[str, str]:
        schema.validate()
        if hashlib.sha256(_canonical_json(schema.to_dict())).hexdigest() != self.schema_hash:
            raise ValueError("Schema does not match the sealed public row domain")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        schema_path = root / "public_schema.json"
        domain_path = root / "public_legal_row_manifest.json"
        schema_path.write_text(
            json.dumps(schema.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        domain_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        return {
            "public_schema": str(schema_path),
            "public_legal_row_manifest": str(domain_path),
            "schema_sha256": self.schema_hash,
            "manifest_sha256": self.manifest_hash,
        }


__all__ = [
    "PUBLIC_LEGAL_ROW_DOMAIN_METHOD",
    "PublicLegalRowDomain",
]
