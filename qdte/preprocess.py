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


def _is_numeric_series(s: pd.Series) -> bool:
    converted = pd.to_numeric(s.dropna(), errors="coerce")
    return len(converted) == len(s.dropna()) and len(converted) > 0


def _sorted_categories(values: pd.Series, missing_token: str) -> list[str]:
    filled = values.astype("string").fillna(missing_token).replace({"": missing_token, "?": missing_token})
    cats = sorted(str(x) for x in filled.unique().tolist())
    return cats


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
    return PreprocessResult(X=X, schema=schema, raw_columns=raw_columns)


def decode_array(X: np.ndarray, schema: TableSchema) -> pd.DataFrame:
    out: dict[str, list[str]] = {}
    for idx, col in enumerate(schema.columns):
        reps = col.representatives or col.categories or [str(i) for i in range(col.cardinality)]
        values = []
        for code in X[:, idx].astype(int).tolist():
            if 0 <= code < len(reps):
                values.append(str(reps[code]))
            else:
                values.append(str(reps[0]))
        out[col.name] = values
    return pd.DataFrame(out)
