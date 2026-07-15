from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.eval.external import write_external_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a canonical external-baseline synthetic table.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--no-block-details", action="store_true")
    parser.add_argument("--true-answers-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_external_evaluation(
        args.input_dir,
        args.synthetic,
        args.output,
        run_metadata_path=args.run_metadata,
        batch_size=args.batch_size,
        include_block_details=not args.no_block_details,
        true_answers_cache_path=args.true_answers_cache,
    )


if __name__ == "__main__":
    main()
