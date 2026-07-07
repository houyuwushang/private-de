#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml
from qdte.dataio import read_json, write_json
from qdte.preprocess import decode_array, load_and_preprocess_csv
from qdte.queries.workload import build_workload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_by_family(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item)] = counts.get(str(item), 0) + 1
    return dict(sorted(counts.items()))


def _workload_summary(qcat: Any, groups: list[Any]) -> dict[str, Any]:
    return {
        "total_queries": int(qcat.m),
        "total_num_queries": int(qcat.m),
        "total_groups": int(len(groups)),
        "num_queries_by_family": _count_by_family(list(qcat.families)),
        "queries_by_family": _count_by_family(list(qcat.families)),
        "num_groups_by_family": _count_by_family([str(group.family) for group in groups]),
        "groups_by_family": _count_by_family([str(group.family) for group in groups]),
        "partition_groups": int(sum(1 for group in groups if bool(group.is_partition))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a canonical external input package directly from a raw CSV/config."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-input-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = args.output_input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_result = load_and_preprocess_csv(config)
    schema = preprocess_result.schema
    X = preprocess_result.X.astype(np.int32, copy=False)
    qcat, groups = build_workload(schema, config)
    summary = _workload_summary(qcat, groups)

    input_csv = Path(config["run"]["input_csv"]).expanduser().resolve()
    shutil.copy2(input_csv, output_dir / "raw.csv")
    np.save(output_dir / "real_encoded.npy", X)
    decode_array(X, schema).to_csv(output_dir / "real_decoded.csv", index=False)
    schema.save_json(output_dir / "schema.json")
    qcat.save_json(output_dir / "queries_full.json")
    write_json([group.to_dict() for group in groups], output_dir / "workload_groups.json")
    write_json(summary, output_dir / "workload_summary.json")

    source_metadata: dict[str, Any] = {}
    if args.source_metadata is not None and args.source_metadata.exists():
        source_metadata = read_json(args.source_metadata)

    metadata = {
        "dataset": str(args.dataset_name),
        "source_dataset": source_metadata.get("dataset", input_csv.stem),
        "config_path": str(args.config.resolve()),
        "source_input_csv": str(input_csv),
        "raw_csv": str((output_dir / "raw.csv").resolve()),
        "n_rows": int(X.shape[0]),
        "n_cols": int(X.shape[1]),
        "cardinalities": [int(col.cardinality) for col in schema.columns],
        "raw_columns": list(preprocess_result.raw_columns),
        "num_queries": int(qcat.m),
        "num_workload_groups": int(len(groups)),
        "groups_by_family": summary["groups_by_family"],
        "queries_by_family": summary["queries_by_family"],
        "preprocess": config.get("preprocess", {}),
        "workload": config.get("workload", {}),
        "privacy_delta_default": float(config.get("privacy", {}).get("delta", 1.0e-9)),
        "schema_sha256": _sha256(output_dir / "schema.json"),
        "queries_sha256": _sha256(output_dir / "queries_full.json"),
        "real_encoded_sha256": _sha256(output_dir / "real_encoded.npy"),
        "raw_sha256": _sha256(output_dir / "raw.csv"),
        "source_metadata": source_metadata,
        "notes": str(args.notes),
    }
    write_json(metadata, output_dir / "metadata.json")
    print(
        f"{args.dataset_name}: rows={X.shape[0]} cols={X.shape[1]} "
        f"queries={qcat.m} groups={len(groups)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
