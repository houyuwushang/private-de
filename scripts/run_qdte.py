#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QDTE synthetic data generation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    if not bool(config.get("runtime", {}).get("xla_preallocate", True)):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from qdte.evolution.engine import run_qdte

    run_qdte(config)


if __name__ == "__main__":
    main()
