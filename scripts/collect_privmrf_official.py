#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from path_defaults import external_results, external_runs
except ModuleNotFoundError:
    from scripts.path_defaults import external_results, external_runs

DEFAULT_RUN_DIR = external_runs() / "privmrf_official" / "official_nltcs_tvd_epsgrid_m300_20260706"
DEFAULT_OUT_PREFIX = external_results() / "privmrf_official_nltcs_tvd_epsgrid_m300_20260706"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _find_result(run_dir: Path, metadata: dict[str, Any]) -> Path:
    copied = metadata.get("copied_result_path")
    if copied and Path(copied).exists():
        return Path(copied)
    candidates = sorted(run_dir.glob("*_TVD.json"))
    if not candidates:
        raise FileNotFoundError(f"no *_TVD.json found under {run_dir}")
    return candidates[0]


def collect(run_dir: Path) -> list[dict[str, Any]]:
    metadata_path = run_dir / "run_metadata.json"
    metadata = _read_json(metadata_path)
    result_path = _find_result(run_dir, metadata)
    result = _read_json(result_path)
    rows: list[dict[str, Any]] = []
    for epsilon, by_dataset in sorted(result.items(), key=lambda item: float(item[0])):
        for dataset, by_way in sorted(by_dataset.items()):
            for way, values in sorted(by_way.items(), key=lambda item: int(item[0])):
                numeric = [float(value) for value in values]
                rows.append(
                    {
                        "method": "PrivMRF official",
                        "task": metadata.get("task", "TVD"),
                        "dataset": dataset,
                        "epsilon": float(epsilon),
                        "way": int(way),
                        "tvd_mean": float(sum(numeric) / len(numeric)),
                        "tvd_values": ";".join(f"{value:.12g}" for value in numeric),
                        "repeat": metadata.get("repeat"),
                        "marginal_num": metadata.get("marginal_num"),
                        "runtime_seconds": metadata.get("runtime_seconds"),
                        "run_dir": str(run_dir),
                        "result_path": str(result_path),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PrivMRF Official TVD Results",
        "",
        "| Dataset | Epsilon | 3-way TVD | 4-way TVD | 5-way TVD |",
        "|---|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, float], dict[int, float]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["epsilon"]))
        grouped.setdefault(key, {})[int(row["way"])] = float(row["tvd_mean"])
    for (dataset, epsilon), by_way in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    f"{epsilon:g}",
                    _fmt(by_way.get(3, math.nan)),
                    _fmt(by_way.get(4, math.nan)),
                    _fmt(by_way.get(5, math.nan)),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PrivMRF official TVD JSON into CSV/Markdown.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args.run_dir.resolve())
    csv_path = args.out_prefix.with_suffix(".csv")
    md_path = args.out_prefix.with_suffix(".md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote markdown to {md_path}")


if __name__ == "__main__":
    main()
