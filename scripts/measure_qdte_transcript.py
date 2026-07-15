#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml, parse_scalar, set_nested
from qdte.measurement.public_artifact import materialize_public_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the private DP measurement stage and seal a public transcript."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a dotted config override; may be repeated.",
    )
    return parser.parse_args()


def _load_config(path: Path, overrides: list[str]) -> dict:
    config = load_yaml(path)
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Override must use KEY=VALUE syntax: {raw!r}")
        key, value = raw.split("=", 1)
        set_nested(config, key, parse_scalar(value))
    return config


def main() -> None:
    args = parse_args()
    verified = materialize_public_transcript(
        _load_config(args.config, args.override),
        args.output_dir,
    )
    privacy = verified.manifest["privacy"]
    print(
        "Sealed public transcript: "
        f"{verified.root} | rows={verified.manifest['public_n_rows']} | "
        f"queries={verified.manifest['num_queries']} | "
        f"epsilon={privacy['epsilon_from_actual_spend']:.12g}"
    )


if __name__ == "__main__":
    main()
