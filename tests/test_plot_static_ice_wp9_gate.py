from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from qdte.dataio import write_json
from scripts.plot_static_ice_wp9_gate import (
    DATASETS,
    METRICS,
    METHOD_ID,
    PROTOCOL_ID,
    build_ratio_rows,
    plot_gate,
    run,
)


def _fixture() -> tuple[list[dict[str, str]], dict]:
    rows: list[dict[str, str]] = []
    cell_ratios: dict[str, float] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        for epsilon in (0.1, 0.3):
            seed_primary: list[float] = []
            for seed in (0, 1, 2):
                aim = {metric: 1.0 + index * 0.1 for index, metric in enumerate(METRICS)}
                factor = 0.7 + dataset_index * 0.1 + seed * 0.01 + epsilon * 0.02
                ice = {metric: value * factor for metric, value in aim.items()}
                seed_primary.append(factor)
                for arm, values in (("static_ice_exact", ice), ("official_aim", aim)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "epsilon": str(epsilon),
                            "seed": str(seed),
                            "arm": arm,
                            **{metric: str(value) for metric, value in values.items()},
                        }
                    )
            cell_ratios[f"{dataset}|{epsilon}"] = math.exp(
                sum(math.log(value) for value in seed_primary) / len(seed_primary)
            )
    gate = {
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "decision": "retain_static_ice_as_development_candidate",
        "cell_primary_ratios": cell_ratios,
    }
    return rows, gate


def test_build_ratio_rows_reproduces_frozen_cell_ratios() -> None:
    rows, gate = _fixture()
    ratios = build_ratio_rows(rows, gate)

    assert len([row for row in ratios if row["level"] == "seed"]) == 24
    aggregate = [row for row in ratios if row["level"] == "aggregate"]
    assert len(aggregate) == 8
    for row in aggregate:
        assert row["primary_ratio"] == pytest.approx(
            gate["cell_primary_ratios"][f"{row['dataset']}|{row['epsilon']}"]
        )


def test_gate_plot_uses_log_axes_and_reference_line() -> None:
    rows, gate = _fixture()
    figure = plot_gate(build_ratio_rows(rows, gate))

    assert len(figure.axes) == 8
    assert all(axis.get_xscale() == "log" for axis in figure.axes)
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    assert all(any(1.0 in line.get_ydata() for line in axis.lines) for axis in figure.axes)
    plt.close(figure)


def test_run_writes_frozen_ratio_figure_and_manifest(tmp_path: Path) -> None:
    rows, gate = _fixture()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    with (input_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(gate, input_dir / "gate_summary.json")

    manifest = run(input_dir, output_dir)

    assert manifest["descriptive_only"] is True
    assert manifest["method_promotion_authorized"] is False
    assert (output_dir / "wp9_static_ice_vs_aim_ratios.csv").is_file()
    assert (output_dir / "wp9_static_ice_vs_aim_gate.pdf").is_file()
    assert (output_dir / "wp9_static_ice_vs_aim_gate.png").is_file()
    assert (output_dir / "figure_manifest.json").is_file()


def test_gate_ratio_mismatch_fails_closed() -> None:
    rows, gate = _fixture()
    gate["cell_primary_ratios"]["adult|0.1"] *= 1.01

    with pytest.raises(RuntimeError, match="disagrees with frozen gate"):
        build_ratio_rows(rows, gate)
