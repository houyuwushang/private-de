#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml
from qdte.preprocess import load_and_preprocess_csv
from qdte.queries.workload import build_workload

try:
    from path_defaults import external_inputs
except ModuleNotFoundError:
    from scripts.path_defaults import external_inputs


DEFAULT_CONFIGS = {
    "adult": "configs/adult_qdte.yaml",
    "acs": "configs/acs_qdte.yaml",
    "br2000": "configs/br2000_qdte.yaml",
    "nltcs": "configs/nltcs_qdte.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write workload_groups.json for canonical external input packages.")
    parser.add_argument("--input-root", type=Path, default=external_inputs())
    parser.add_argument("--datasets", default="adult,acs,br2000,nltcs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for dataset in datasets:
        if dataset not in DEFAULT_CONFIGS:
            raise ValueError(f"No default config is registered for dataset {dataset!r}")
        cfg = load_yaml(DEFAULT_CONFIGS[dataset])
        pp = load_and_preprocess_csv(cfg)
        _, groups = build_workload(pp.schema, cfg)
        output_path = args.input_root / dataset / "workload_groups.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(orjson.dumps([group.to_dict() for group in groups], option=orjson.OPT_INDENT_2))
        partition_blocks = sum(1 for group in groups if group.is_partition and len(group.query_indices) > 1)
        print(
            f"{dataset}: groups={len(groups)} partition_blocks={partition_blocks} "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
