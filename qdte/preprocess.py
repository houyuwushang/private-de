from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qdte.schema import ColumnSchema, TableSchema


@dataclass
class PreprocessResult:
    X: np.ndarray
    schema: TableSchema
    raw_columns: list[str]
    public_schema_path: Path | None = None


def _is_numeric_series(s: pd.Series) -> bool:
    converted = pd.to_numeric(s.dropna(), errors="coerce")
    return len(converted) == len(s.dropna()) and len(converted) > 0


def _sorted_categories(values: pd.Series, missing_token: str) -> list[str]:
    filled = values.astype("string").fillna(missing_token).replace({"": missing_token, "?": missing_token})
    cats = sorted(str(x) for x in filled.unique().tolist())
    return cats


def _normalized_strings(values: pd.Series, missing_token: str) -> pd.Series:
    return values.astype("string").fillna(missing_token).replace({"": missing_token, "?": missing_token})


def _encode_public_codebook(
    values: pd.Series,
    column: ColumnSchema,
) -> np.ndarray:
    labels = column.categories or column.representatives
    if labels is None:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Public schema column {column.name!r} has no codebook and requires integer encoded input"
            )
        array = numeric.to_numpy(dtype=np.float64)
        rounded = np.rint(array)
        if not np.allclose(array, rounded, atol=0.0, rtol=0.0):
            raise ValueError(f"Public schema column {column.name!r} contains non-integer encoded values")
        encoded = rounded.astype(np.int32)
        if np.any(encoded < 0) or np.any(encoded >= int(column.cardinality)):
            raise ValueError(f"Public schema column {column.name!r} contains values outside its public domain")
        return encoded

    mapping = {str(label): idx for idx, label in enumerate(labels)}
    normalized = _normalized_strings(values, column.missing_token)
    observed = [str(value) for value in normalized.tolist()]
    unknown = sorted(set(observed) - set(mapping))
    if unknown:
        preview = unknown[:5]
        raise ValueError(
            f"Public schema column {column.name!r} contains values absent from its public codebook: {preview}"
        )
    return np.asarray([mapping[value] for value in observed], dtype=np.int32)


def _encode_public_numeric(values: pd.Series, column: ColumnSchema) -> np.ndarray:
    edges = np.asarray(column.bin_edges or [], dtype=np.float64)
    if edges.size < 2:
        return _encode_public_codebook(values, column)

    normalized = _normalized_strings(values, column.missing_token)
    missing = normalized == column.missing_token
    numeric = pd.to_numeric(normalized.mask(missing), errors="coerce")
    invalid = numeric.isna() & ~missing
    if invalid.any():
        bad = sorted(set(str(value) for value in normalized[invalid].tolist()))[:5]
        raise ValueError(f"Public numeric schema column {column.name!r} contains non-numeric values: {bad}")

    num_bins = int(edges.size - 1)
    if int(column.cardinality) not in {num_bins, num_bins + 1}:
        raise ValueError(
            f"Public numeric schema column {column.name!r} cardinality must equal its bin count "
            "or bin count plus one missing category"
        )
    if missing.any() and int(column.cardinality) != num_bins + 1:
        raise ValueError(
            f"Public numeric schema column {column.name!r} has missing values but no public missing category"
        )

    array = numeric.fillna(float(edges[0])).to_numpy(dtype=np.float64)
    encoded = np.searchsorted(edges[1:-1], array, side="right").astype(np.int32)
    if missing.any():
        encoded[missing.to_numpy()] = num_bins
    return encoded


def _encode_with_public_schema(df: pd.DataFrame, schema: TableSchema) -> np.ndarray:
    schema.validate()
    raw_columns = [str(column) for column in df.columns.tolist()]
    schema_columns = [str(column.name) for column in schema.columns]
    if raw_columns != schema_columns:
        raise ValueError(
            "Input CSV columns must exactly match the ordered public schema columns; "
            f"input={raw_columns}, schema={schema_columns}"
        )

    encoded_columns: list[np.ndarray] = []
    for column in schema.columns:
        values = df[str(column.name)]
        if column.kind == "categorical":
            encoded = _encode_public_codebook(values, column)
        elif column.kind == "numerical_binned":
            encoded = _encode_public_numeric(values, column)
        else:
            raise ValueError(f"Unsupported public schema kind {column.kind!r}")
        encoded_columns.append(encoded)
    return np.stack(encoded_columns, axis=1).astype(np.int32)


def _public_schema_path(input_csv: Path, configured: Any) -> Path | None:
    if configured is None:
        return None
    text = str(configured).strip()
    if not text:
        raise ValueError("preprocess.public_schema_json must be a non-empty path")
    if text in {"sibling", "input_sibling"}:
        return input_csv.parent / "schema.json"
    return Path(text).expanduser()


def _encode_categorical(values: pd.Series, name: str, missing_token: str) -> tuple[np.ndarray, ColumnSchema]:
    filled = values.astype("string").fillna(missing_token).replace({"": missing_token, "?": missing_token})
    cats = _sorted_categories(filled, missing_token)
    mapping = {cat: idx for idx, cat in enumerate(cats)}
    encoded = filled.map(lambda x: mapping[str(x)]).to_numpy(dtype=np.int32)
    schema = ColumnSchema(
        name=name,
        kind="categorical",
        cardinality=len(cats),
        categories=cats,
        representatives=cats,
        missing_token=missing_token,
    )
    return encoded, schema


def _encode_numeric_binned(
    values: pd.Series,
    name: str,
    missing_token: str,
    numerical_bins: int,
) -> tuple[np.ndarray, ColumnSchema]:
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna()
    non_missing = numeric[~missing]
    if non_missing.empty:
        encoded = np.zeros(len(values), dtype=np.int32)
        schema = ColumnSchema(
            name=name,
            kind="numerical_binned",
            cardinality=1,
            bin_edges=[],
            representatives=[missing_token],
            missing_token=missing_token,
        )
        return encoded, schema

    unique_vals = np.sort(non_missing.unique())
    if len(unique_vals) <= numerical_bins:
        reps = [str(x) for x in unique_vals.tolist()]
        mapping = {float(v): i for i, v in enumerate(unique_vals.tolist())}
        encoded = numeric.map(lambda x: mapping.get(float(x), len(reps))).to_numpy(dtype=np.int32)
        if missing.any():
            reps.append(missing_token)
        else:
            encoded[missing.to_numpy()] = 0
        schema = ColumnSchema(
            name=name,
            kind="numerical_binned",
            cardinality=len(reps),
            bin_edges=[],
            representatives=reps,
            missing_token=missing_token,
        )
        return encoded, schema

    quantiles = np.linspace(0.0, 1.0, numerical_bins + 1)
    edges = np.unique(np.quantile(non_missing.to_numpy(dtype=float), quantiles))
    if len(edges) <= 2:
        edges = np.asarray([float(non_missing.min()), float(non_missing.max())], dtype=float)
    inner_edges = edges[1:-1]
    encoded = np.searchsorted(inner_edges, numeric.to_numpy(dtype=float), side="right").astype(np.int32)
    reps: list[str] = []
    for b in range(len(edges) - 1):
        lo = edges[b]
        hi = edges[b + 1]
        mask = (non_missing >= lo) & (non_missing <= hi if b == len(edges) - 2 else non_missing < hi)
        if mask.any():
            reps.append(str(float(non_missing[mask].median())))
        else:
            reps.append(str(float((lo + hi) / 2.0)))
    if missing.any():
        missing_id = len(reps)
        encoded[missing.to_numpy()] = missing_id
        reps.append(missing_token)
    schema = ColumnSchema(
        name=name,
        kind="numerical_binned",
        cardinality=len(reps),
        bin_edges=[float(x) for x in edges.tolist()],
        representatives=reps,
        missing_token=missing_token,
    )
    return encoded, schema


def load_and_preprocess_csv(config: dict[str, Any]) -> PreprocessResult:
    run_cfg = config.get("run", {})
    pp_cfg = config.get("preprocess", {})
    privacy_cfg = config.get("privacy", {})
    input_csv = Path(run_cfg["input_csv"]).expanduser()
    missing_token = str(pp_cfg.get("missing_token", "__MISSING__"))
    numerical_bins = int(pp_cfg.get("numerical_bins", 32))
    label_column = pp_cfg.get("label_column")
    numerical_columns = set(str(x) for x in pp_cfg.get("numerical_columns", []))
    categorical_columns = set(str(x) for x in pp_cfg.get("categorical_columns", []))
    auto_numeric_min_unique = int(pp_cfg.get("auto_numeric_min_unique", 10))
    force_all_categorical = bool(pp_cfg.get("force_all_categorical", False))

    df = pd.read_csv(input_csv)
    raw_columns = [str(c) for c in df.columns.tolist()]
    df.columns = raw_columns
    if len(df) == 0 or len(raw_columns) == 0:
        raise ValueError("Input CSV must contain at least one row and one column")

    public_schema_path = _public_schema_path(input_csv, pp_cfg.get("public_schema_json"))
    dp_release_mode = bool(privacy_cfg.get("dp_release_mode", False))
    if dp_release_mode and str(privacy_cfg.get("mode", "dp")).lower() == "dp" and public_schema_path is None:
        raise ValueError("privacy.dp_release_mode=true requires preprocess.public_schema_json")
    if public_schema_path is not None:
        if not public_schema_path.is_file():
            raise FileNotFoundError(f"Public schema file does not exist: {public_schema_path}")
        schema = TableSchema.load_json(public_schema_path)
        X = _encode_with_public_schema(df, schema)
        return PreprocessResult(
            X=X,
            schema=schema,
            raw_columns=raw_columns,
            public_schema_path=public_schema_path.resolve(),
        )

    configured_columns = numerical_columns | categorical_columns
    unknown_columns = sorted(configured_columns - set(raw_columns))
    if unknown_columns:
        raise ValueError(f"Configured preprocess columns are missing from the CSV: {unknown_columns}")
    overlapping_columns = sorted(numerical_columns & categorical_columns)
    if overlapping_columns:
        raise ValueError(f"Columns cannot be both numerical and categorical: {overlapping_columns}")
    if label_column is not None and str(label_column) not in raw_columns:
        raise ValueError(f"preprocess.label_column {label_column!r} is missing from the CSV")
    if numerical_bins <= 0:
        raise ValueError("preprocess.numerical_bins must be positive")
    if auto_numeric_min_unique <= 0:
        raise ValueError("preprocess.auto_numeric_min_unique must be positive")

    encoded_cols: list[np.ndarray] = []
    schema_cols: list[ColumnSchema] = []
    for name in raw_columns:
        series = df[name]
        if name in categorical_columns or force_all_categorical:
            encoded, col_schema = _encode_categorical(series, name, missing_token)
        else:
            numeric_like = _is_numeric_series(series)
            unique_count = int(series.nunique(dropna=True))
            should_numeric = name in numerical_columns or (numeric_like and unique_count >= auto_numeric_min_unique)
            if should_numeric:
                encoded, col_schema = _encode_numeric_binned(series, name, missing_token, numerical_bins)
            else:
                encoded, col_schema = _encode_categorical(series, name, missing_token)
        encoded_cols.append(encoded)
        schema_cols.append(col_schema)

    X = np.stack(encoded_cols, axis=1).astype(np.int32)
    schema = TableSchema(columns=schema_cols, label_column=str(label_column) if label_column is not None else None)
    schema.validate()
    return PreprocessResult(X=X, schema=schema, raw_columns=raw_columns)


def decode_array(X: np.ndarray, schema: TableSchema) -> pd.DataFrame:
    schema.validate()
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != schema.d or not np.issubdtype(X.dtype, np.integer):
        raise ValueError(f"X must be a two-dimensional integer table with {schema.d} columns")
    out: dict[str, list[str]] = {}
    for idx, col in enumerate(schema.columns):
        reps = col.representatives or col.categories or [str(i) for i in range(col.cardinality)]
        values = []
        for code in X[:, idx].astype(int).tolist():
            if not 0 <= code < int(col.cardinality):
                raise ValueError(f"Encoded value {code} is outside the public domain for column {col.name!r}")
            values.append(str(reps[code]))
        out[col.name] = values
    return pd.DataFrame(out)
