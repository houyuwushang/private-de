from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_qdte_paper_results.py"
    spec = importlib.util.spec_from_file_location("plot_qdte_paper_results", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_figures_writes_pdf_and_png(tmp_path: Path) -> None:
    module = _load_module()
    curve_rows = []
    endpoint_rows = []
    for dataset in module.DATASET_ORDER:
        curve_rows.extend(
            [
                {
                    "dataset_label": dataset,
                    "time_seconds": 1.0,
                    "loss": 20.0,
                    "gsd_final_loss": 10.0,
                    "gsd_runtime_seconds": 10.0,
                },
                {
                    "dataset_label": dataset,
                    "time_seconds": 5.0,
                    "loss": 8.0,
                    "gsd_final_loss": 10.0,
                    "gsd_runtime_seconds": 10.0,
                },
            ]
        )
        endpoint_rows.append(
            {"dataset_label": dataset, "time_to_first_pass_over_gsd": 0.5}
        )
    _write_csv(tmp_path / "tables/structured_vs_gsd_time_curve.csv", curve_rows)
    _write_csv(tmp_path / "tables/structured_vs_gsd_endpoint.csv", endpoint_rows)
    (tmp_path / "tables/transfer_gap_diagnostic.json").write_text(
        json.dumps(
            {
                "transfer_decomposition": {
                    "relative_vs_baseline_pct": {
                        "T_target_to_true_norm2": -1.2,
                        "F_synthetic_to_target_norm2": -1.6,
                        "E_synthetic_to_true_norm2": 0.35,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = module.build_figures(tmp_path)

    assert len(outputs) == 4
    assert all(path.exists() and path.stat().st_size > 100 for path in outputs)
