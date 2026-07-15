#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.preprocess import decode_array
from qdte.schema import TableSchema


def _normalize_token(value: object, missing_token: str) -> str:
    if pd.isna(value):
        return missing_token
    text = str(value)
    if text == "" or text == "?":
        return missing_token
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _encode_categorical(series: pd.Series, categories: list[str], missing_token: str, name: str) -> np.ndarray:
    mapping = {str(cat): idx for idx, cat in enumerate(categories)}
    values: list[int] = []
    unknown: set[str] = set()
    for value in series.tolist():
        token = _normalize_token(value, missing_token)
        if token not in mapping:
            unknown.add(token)
            token = missing_token
        if token not in mapping:
            raise ValueError(f"Column {name!r} has values outside schema categories: {sorted(unknown)[:5]}")
        values.append(int(mapping[token]))
    return np.asarray(values, dtype=np.int32)


def _encode_numeric(series: pd.Series, representatives: list[str], bin_edges: list[float], missing_token: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna().to_numpy()
    if bin_edges:
        edges = np.asarray(bin_edges, dtype=np.float64)
        inner_edges = edges[1:-1]
        encoded = np.searchsorted(inner_edges, numeric.to_numpy(dtype=np.float64), side="right").astype(np.int32)
        if missing.any():
            missing_idx = representatives.index(missing_token) if missing_token in representatives else len(representatives) - 1
            encoded[missing] = int(missing_idx)
        return np.clip(encoded, 0, len(representatives) - 1).astype(np.int32, copy=False)

    rep_values: list[float] = []
    rep_indices: list[int] = []
    missing_idx = representatives.index(missing_token) if missing_token in representatives else None
    for idx, rep in enumerate(representatives):
        if rep == missing_token:
            continue
        try:
            rep_values.append(float(rep))
            rep_indices.append(int(idx))
        except ValueError:
            continue
    if not rep_values:
        fill = 0 if missing_idx is None else int(missing_idx)
        return np.full(series.shape[0], fill, dtype=np.int32)

    reps = np.asarray(rep_values, dtype=np.float64)
    rep_idx = np.asarray(rep_indices, dtype=np.int32)
    values = numeric.to_numpy(dtype=np.float64)
    nearest = np.argmin(np.abs(values[:, None] - reps[None, :]), axis=1)
    encoded = rep_idx[nearest].astype(np.int32, copy=False)
    if missing.any():
        encoded[missing] = int(missing_idx if missing_idx is not None else rep_idx[0])
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a raw synthetic CSV using a canonical external schema.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decoded-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = TableSchema.load_json(args.input_dir / "schema.json")
    df = pd.read_csv(args.csv)
    missing = [col.name for col in schema.columns if col.name not in df.columns]
    if missing:
        raise ValueError(f"Synthetic CSV is missing schema columns: {missing}")

    encoded_cols: list[np.ndarray] = []
    for col in schema.columns:
        series = df[col.name]
        if col.kind == "categorical":
            encoded = _encode_categorical(series, col.categories or [], col.missing_token, col.name)
        elif col.kind == "numerical_binned":
            encoded = _encode_numeric(series, col.representatives or [], col.bin_edges or [], col.missing_token)
        else:
            raise ValueError(f"Unknown schema column kind {col.kind!r}")
        encoded_cols.append(encoded)

    X = np.stack(encoded_cols, axis=1).astype(np.int32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, X)
    if args.decoded_output is not None:
        args.decoded_output.parent.mkdir(parents=True, exist_ok=True)
        decode_array(X, schema).to_csv(args.decoded_output, index=False)
    print(f"encoded {args.csv} -> {args.output} shape={X.shape}")


if __name__ == "__main__":
    main()
