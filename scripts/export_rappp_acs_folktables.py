#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from path_defaults import external_inputs, rappp_root
except ModuleNotFoundError:
    from scripts.path_defaults import external_inputs, rappp_root

RAPPP_REPO = rappp_root()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the upstream RAP++ ACS/Folktables task split as CSV files."
    )
    parser.add_argument("--rappp-root", type=Path, default=RAPPP_REPO)
    parser.add_argument("--state", default="CA")
    parser.add_argument("--target", default="income")
    parser.add_argument("--survey-year", type=int, default=2014)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=external_inputs() / "acs_ca_income_rappp_official_source",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rappp_root = args.rappp_root.resolve()
    if str(rappp_root) not in sys.path:
        sys.path.insert(0, str(rappp_root))

    from dataloading.data_functions.acs import get_acs

    start = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_fn = get_acs(state=str(args.state), target=str(args.target), survey_year=int(args.survey_year))
    container = dataset_fn(int(args.seed))
    train_df = container.from_dataset_to_df_fn(container.train)
    test_df = container.from_dataset_to_df_fn(container.test)

    train_path = output_dir / "raw.csv"
    test_path = output_dir / "test_raw.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "dataset": f"acs_{args.state}_{args.target}_rappp_official",
        "source": "RAP++ upstream Folktables ACS loader",
        "rappp_root": str(rappp_root),
        "state": str(args.state),
        "target": str(args.target),
        "survey_year": int(args.survey_year),
        "seed": int(args.seed),
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "columns": [str(col) for col in train_df.columns.tolist()],
        "categorical_columns": [str(col) for col in container.cat_columns],
        "numerical_columns": [str(col) for col in container.num_columns],
        "label_columns": [str(col) for col in container.label_column],
        "domain_attrs": [str(col) for col in container.train.domain.attrs],
        "domain_shape": [int(x) for x in container.train.domain.shape],
        "domain_onehot_dim": int(sum(container.train.domain.shape)),
        "output_dir": str(output_dir),
        "raw_csv": str(train_path),
        "test_raw_csv": str(test_path),
        "runtime_seconds": float(time.time() - start),
        "notes": (
            "raw.csv is the upstream RAP++ train split after its ACS transformer "
            "inverse transform; test_raw.csv is exported for optional downstream ML checks."
        ),
    }
    _write_json(output_dir / "source_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
