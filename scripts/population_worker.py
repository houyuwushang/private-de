#!/usr/bin/env python
from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import load_yaml


def main() -> None:
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") is None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from qdte.evolution.engine import run_qdte

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        log_path = Path(str(task["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config = load_yaml(task["config_path"])
            with log_path.open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    print(
                        "Persistent population worker "
                        f"slot={os.environ.get('QDTE_POPULATION_WORKER_SLOT', '')} "
                        f"visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                        flush=True,
                    )
                    run_qdte(config)
            response = {"ok": True}
        except BaseException:
            response = {"ok": False, "error": traceback.format_exc()}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
