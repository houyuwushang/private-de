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
from qdte.queries.workload import build_workload
from qdte.schema import TableSchema


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_required_inputs(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw.csv", "real_encoded.npy", "real_decoded.csv", "schema.json"):
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required source input file: {src}")
        shutil.copy2(src, output_dir / name)


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
    parser = argparse.ArgumentParser(description="Create a canonical external input package with a workload profile.")
    parser.add_argument("--source-input-dir", required=True, type=Path)
    parser.add_argument("--output-input-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_input_dir
    output_dir = args.output_input_dir
    _copy_required_inputs(source_dir, output_dir)

    config = load_yaml(args.config)
    schema = TableSchema.load_json(output_dir / "schema.json")
    qcat, groups = build_workload(schema, config)
    qcat.save_json(output_dir / "queries_full.json")
    write_json([group.to_dict() for group in groups], output_dir / "workload_groups.json")
    summary = _workload_summary(qcat, groups)
    write_json(summary, output_dir / "workload_summary.json")

    source_metadata = read_json(source_dir / "metadata.json") if (source_dir / "metadata.json").exists() else {}
    real = np.load(output_dir / "real_encoded.npy")
    metadata = {
        "dataset": str(args.dataset_name),
        "source_dataset": source_metadata.get("dataset", source_dir.name),
        "config_path": str(args.config.resolve()),
        "source_input_dir": str(source_dir.resolve()),
        "raw_csv": str((output_dir / "raw.csv").resolve()),
        "n_rows": int(real.shape[0]),
        "n_cols": int(real.shape[1]),
        "cardinalities": [int(col.cardinality) for col in schema.columns],
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
        "notes": str(args.notes),
    }
    write_json(metadata, output_dir / "metadata.json")
    print(
        f"{args.dataset_name}: queries={qcat.m} groups={len(groups)} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
