#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml, set_nested
from qdte.evolution.engine import run_qdte


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    variant = args.variant
    out = Path(config.get("run", {}).get("output_dir", "outputs/qdte_run"))
    set_nested(config, "run.output_dir", str(out.parent / f"{out.name}_{variant}"))
    if variant == "random_mutation":
        set_nested(config, "qdte.num_active_targets", 0)
        set_nested(config, "qdte.random_candidate_fraction", 1.0)
    elif variant == "no_edit_cost":
        set_nested(config, "qdte.lambda_cost", 0.0)
    elif variant == "sequential_greedy":
        set_nested(config, "qdte.accepted_per_iter", 1)
        set_nested(config, "qdte.transport_mode", "sequential_greedy")
    elif variant == "target_only":
        set_nested(config, "qdte.score_backend", "target_only")
    elif variant == "no_threshold":
        set_nested(config, "qdte.kappa_noise", 0.0)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    run_qdte(config)


if __name__ == "__main__":
    main()
