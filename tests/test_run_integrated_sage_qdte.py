from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from qdte.dataio import read_json
from qdte.measurement.measure import MeasurementGroup, Measurements, measurements_from_public_dict
from qdte.queries.types import OP_EQ, OP_LE, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_final_protocol_request_is_exactly_frozen() -> None:
    module = _load_script("run_integrated_sage_qdte")

    module.validate_protocol_request(
        protocol_mode="final",
        epsilon=1.0,
        delta=1.0e-9,
        rounds=50,
        inner_iters=5000,
        final_refit_iters=5000,
        projection_profile="P1_lightweight",
    )

    with pytest.raises(ValueError, match="50/5000/5000"):
        module.validate_protocol_request(
            protocol_mode="final",
            epsilon=1.0,
            delta=1.0e-9,
            rounds=50,
            inner_iters=20,
            final_refit_iters=100,
            projection_profile="P1_lightweight",
        )
    with pytest.raises(ValueError, match="P1_lightweight"):
        module.validate_protocol_request(
            protocol_mode="final",
            epsilon=1.0,
            delta=1.0e-9,
            rounds=50,
            inner_iters=5000,
            final_refit_iters=5000,
            projection_profile="P3_nonnegative",
        )


def test_integrated_config_preserves_qdte_standard_profile(tmp_path: Path) -> None:
    module = _load_script("run_integrated_sage_qdte")
    base = module.load_yaml(ROOT / "configs" / "adult_sage_strong.yaml")
    original = copy.deepcopy(base)

    config = module.build_integrated_config(
        base,
        seed=2,
        output_dir=tmp_path,
        rho_total=module.rho_for_epsilon(1.0, 1.0e-9),
        delta=1.0e-9,
        projection_profile="P1_lightweight",
        input_csv_override=tmp_path / "public_input.csv",
    )

    assert base == original
    assert config["privacy"]["mode"] == "dp"
    assert config["qdte"]["objective_weighting"] == "variance"
    assert config["qdte"]["transport_mode"] == "atom_flow"
    assert config["projection"]["consistency"]["enabled"] is False
    assert config["run"]["input_csv"] == str(tmp_path / "public_input.csv")


def test_nonnegative_profile_enables_certified_projection(tmp_path: Path) -> None:
    module = _load_script("run_integrated_sage_qdte")
    base = module.load_yaml(ROOT / "configs" / "nltcs_sage_strong.yaml")

    config = module.build_integrated_config(
        base,
        seed=0,
        output_dir=tmp_path,
        rho_total=module.rho_for_epsilon(10.0, 1.0e-9),
        delta=1.0e-9,
        projection_profile="P3_nonnegative",
    )

    consistency = config["projection"]["consistency"]
    assert consistency["enabled"] is True
    assert consistency["method"] == "query_space_feasible_lsq"
    assert consistency["certificate_max_iterations"] == 1000


def test_integrated_runner_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    module = _load_script("run_integrated_sage_qdte")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "old.txt").write_text("old evidence\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new or empty"):
        module.run_integrated(
            config_path=ROOT / "configs" / "adult_sage_strong.yaml",
            output_dir=output,
            seed=0,
            epsilon=10.0,
            delta=1.0e-9,
            protocol_mode="smoke",
            rounds=1,
            inner_iters=0,
            final_refit_iters=0,
            projection_profile="P1_lightweight",
        )


def test_adaptive_generator_profiles_do_not_silently_mix() -> None:
    module = _load_script("run_adaptive_selection_ablation")
    base = {
        "privacy": {"mode": "dp", "measurement_mode": "static_all"},
        "qdte": {
            "objective_weighting": "variance",
            "transport_mode": "atom_flow",
        },
    }

    standard = module._configure_a_generator(
        base,
        output_dir=Path("out"),
        init_path=Path("init.npy"),
        measurement_dir=Path("measurement"),
        inner_iters=20,
        generator_profile="qdte_standard",
    )
    legacy = module._configure_a_generator(
        base,
        output_dir=Path("out"),
        init_path=Path("init.npy"),
        measurement_dir=Path("measurement"),
        inner_iters=20,
        generator_profile="adaptive_legacy",
    )
    fresh = module._configure_a_generator(
        base,
        output_dir=Path("out"),
        init_path=Path("init.npy"),
        measurement_dir=Path("measurement"),
        inner_iters=5000,
        generator_profile="qdte_standard",
        generator_seed=12345,
        stop_patience=500,
    )

    assert standard["qdte"]["transport_mode"] == "atom_flow"
    assert legacy["qdte"]["transport_mode"] == "constructive_pair"
    assert standard["qdte"]["max_iters"] == 20
    assert standard["evaluation"]["compute_true_query_error"] is False
    assert fresh["run"]["seed"] == 12345
    assert fresh["qdte"]["max_iters"] == 5000
    assert fresh["qdte"]["stop_patience"] == 500


def test_adaptive_generator_stage_seeds_are_fresh_and_reproducible() -> None:
    module = _load_script("run_adaptive_selection_ablation")

    first = module._generator_stage_seed(7, 1)
    assert first == module._generator_stage_seed(7, 1)
    assert first != module._generator_stage_seed(7, 2)
    assert first != module._generator_stage_seed(8, 1)

    with pytest.raises(ValueError, match="stage_id"):
        module._generator_stage_seed(7, 0)


def test_aim_style_measurement_plan_anneals_and_exhausts_budget() -> None:
    module = _load_script("run_adaptive_selection_ablation")

    regular = module._plan_annealed_measurement_round(
        current_sigma=10.0,
        remaining_rho=0.02,
        remaining_rounds=5,
    )
    assert np.isclose(regular.rho, 0.005)
    assert np.isclose(regular.sigma, 10.0)
    assert regular.exhausts_budget is False

    final = module._plan_annealed_measurement_round(
        current_sigma=5.0,
        remaining_rho=0.03,
        remaining_rounds=4,
    )
    assert np.isclose(final.rho, 0.03)
    assert np.isclose(final.sigma, math.sqrt(1.0 / 0.06))
    assert final.exhausts_budget is True


def test_aim_style_private_round_accounts_em_and_measurement_together() -> None:
    module = _load_script("run_adaptive_selection_ablation")

    regular = module._plan_annealed_private_round(
        current_measurement_sigma=10.0,
        current_selection_epsilon=0.2,
        remaining_rho=0.1,
        remaining_rounds=5,
        charge_selection=True,
    )
    assert np.isclose(regular.measurement_rho, 0.005)
    assert np.isclose(regular.selection_rho, 0.005)
    assert np.isclose(regular.measurement_sigma, 10.0)
    assert np.isclose(regular.selection_epsilon, 0.2)
    assert regular.exhausts_budget is False

    final = module._plan_annealed_private_round(
        current_measurement_sigma=10.0,
        current_selection_epsilon=0.2,
        remaining_rho=0.015,
        remaining_rounds=4,
        charge_selection=True,
    )
    assert np.isclose(final.measurement_rho, 0.0075)
    assert np.isclose(final.selection_rho, 0.0075)
    assert np.isclose(final.measurement_rho + final.selection_rho, 0.015)
    assert final.exhausts_budget is True


def test_released_model_change_anneal_signal_uses_only_synthetic_answers() -> None:
    module = _load_script("run_adaptive_selection_ablation")
    before = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    after = np.asarray([10.2, 19.8, 100.0], dtype=np.float64)

    change, floor, anneal = module._released_model_change_anneal_diagnostic(
        before_answers=before,
        after_answers=after,
        query_indices=np.asarray([0, 1], dtype=np.int32),
        noise_std=1.0,
    )

    assert np.isclose(change, 0.4)
    assert np.isclose(floor, 2.0 * math.sqrt(2.0 / math.pi))
    assert anneal is True


def test_private_sage_selector_has_registered_unit_sensitivity() -> None:
    module = _load_script("run_adaptive_selection_ablation")

    assert np.isclose(
        module._certified_private_selection_sensitivity(
            "voi_sageordergain_harmonic_qproject_repeat"
        ),
        1.0,
    )
    assert np.isclose(
        module._certified_private_selection_sensitivity("aim_l1_floor_repeat"),
        1.0,
    )
    with pytest.raises(ValueError, match="no registered"):
        module._certified_private_selection_sensitivity("uncertified_score")


def test_exponential_sampler_respects_explicit_sensitivity() -> None:
    module = _load_script("run_adaptive_selection_ablation")
    scores = np.asarray([0.0, 2.0], dtype=np.float64)
    seed = 123

    actual = module._sample_exponential(
        scores,
        epsilon=1.0,
        rng=np.random.default_rng(seed),
        sensitivity=2.0,
    )
    expected = module.sample_exponential_mechanism(
        scores,
        epsilon=1.0,
        sensitivity=2.0,
        rng=np.random.default_rng(seed),
    )
    assert actual == expected


def test_released_sage_rank_prior_is_bounded_and_tie_stable() -> None:
    module = _load_script("run_adaptive_selection_ablation")

    log_weights = module._rank_log_base_measure(
        np.asarray([0.0, 1.0, 1.0, 3.0], dtype=np.float64),
        max_odds=4.0,
    )

    assert np.isclose(np.max(log_weights), 0.0)
    assert np.isclose(np.min(log_weights), -math.log(4.0))
    assert np.isclose(log_weights[1], log_weights[2])
    assert np.exp(np.max(log_weights) - np.min(log_weights)) <= 4.0 + 1.0e-12


def test_released_sage_base_measure_does_not_change_private_sensitivity() -> None:
    module = _load_script("run_adaptive_selection_ablation")
    private = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    neighbor = np.asarray([9.5, 20.5, 29.0], dtype=np.float64)
    released = np.asarray([3.0, 1.0, 2.0], dtype=np.float64)

    adjusted, log_prior = module._apply_released_score_base_measure(
        private,
        released,
        epsilon=0.1,
        sensitivity=1.0,
        max_odds=4.0,
    )
    neighbor_adjusted, neighbor_log_prior = module._apply_released_score_base_measure(
        neighbor,
        released,
        epsilon=0.1,
        sensitivity=1.0,
        max_odds=4.0,
    )

    assert np.allclose(log_prior, neighbor_log_prior)
    assert np.allclose(adjusted - neighbor_adjusted, private - neighbor)
    assert np.allclose(0.05 * adjusted, 0.05 * private + log_prior)


def test_adaptive_measurement_artifact_is_self_contained(tmp_path: Path) -> None:
    module = _load_script("run_adaptive_selection_ablation")
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "x=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "x=1", "oneway:0", "oneway")
    builder.add([(0, OP_LE, 0, 0, 0)], "x<=0", "prefix:0", "prefix")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name="x",
                kind="categorical",
                cardinality=2,
                categories=["0", "1"],
            )
        ]
    )
    block = module.AdaptiveBlock(
        name="oneway:0",
        family="oneway",
        query_indices=np.asarray([0, 1], dtype=np.int32),
        delta_l2=1.0,
        is_vector=True,
        scope=(0,),
        coverage_weight=2.0,
    )
    record = module.MeasurementRecord(
        block_id=0,
        noisy=np.asarray([6.0, 4.0], dtype=np.float64),
    )

    artifact = module._measurement_artifact(
        qcat=qcat,
        schema=schema,
        blocks=[block],
        measurement_records=[record],
        current_syn_answers=np.asarray([5.0, 5.0, 5.0], dtype=np.float64),
        total_rows=10,
        cardinalities=np.asarray([2], dtype=np.int32),
        projection_cfg={"project_partitions": True, "clip_nonpartition": True},
        output_dir=tmp_path / "measurement",
        epsilon=1.0,
        measurement_sigma=1.0,
        delta=1.0e-9,
        selection_rho_per_round=0.0,
    )

    assert (artifact / "queries.json").is_file()
    assert (artifact / "schema.json").is_file()
    assert (artifact / "measurements.json").is_file()
    assert TableSchema.load_json(artifact / "schema.json").to_dict() == schema.to_dict()
    loaded = measurements_from_public_dict(read_json(artifact / "measurements.json"))
    assert loaded.num_rows == 10
    assert any(group.name == "adaptive:unmeasured" for group in loaded.groups)


def test_coverage_and_adaptive_measurements_use_precision_averaging(tmp_path: Path) -> None:
    module = _load_script("run_adaptive_selection_ablation")
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "x=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "x=1", "oneway:0", "oneway")
    builder.add([(0, OP_LE, 0, 0, 0)], "x<=0", "prefix:0", "prefix")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema(
                name="x",
                kind="categorical",
                cardinality=2,
                categories=["0", "1"],
            )
        ]
    )
    groups = [
        MeasurementGroup(
            query_indices=np.asarray([0, 1], dtype=np.int32),
            sensitivity_l2=1.0,
            rho=0.25,
            sigma=2.0,
            noise_std=2.0,
            name="oneway:0",
            family="oneway",
            is_partition=True,
        ),
        MeasurementGroup(
            query_indices=np.asarray([2], dtype=np.int32),
            sensitivity_l2=1.0,
            rho=0.25,
            sigma=2.0,
            noise_std=2.0,
            name="prefix:0",
            family="prefix",
            is_partition=False,
        ),
    ]
    base = Measurements(
        target_noisy=np.asarray([6.0, 4.0, 6.0], dtype=np.float32),
        target_projected=np.asarray([6.0, 4.0, 6.0], dtype=np.float32),
        variances=np.asarray([4.0, 4.0, 4.0], dtype=np.float32),
        inv_variances=np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
        groups=groups,
        mode="dp",
        rho_total=0.5,
        rho_spent=0.5,
        epsilon_delta=1.0,
        delta=1.0e-9,
        num_rows=10,
    )
    block = module.AdaptiveBlock(
        name="oneway:0",
        family="oneway",
        query_indices=np.asarray([0, 1], dtype=np.int32),
        delta_l2=1.0,
        is_vector=True,
        scope=(0,),
        coverage_weight=1.0,
    )
    record = module.MeasurementRecord(
        block_id=0,
        noisy=np.asarray([8.0, 2.0], dtype=np.float64),
    )

    artifact = module._measurement_artifact(
        qcat=qcat,
        schema=schema,
        blocks=[block],
        measurement_records=[record],
        current_syn_answers=np.asarray([5.0, 5.0, 5.0], dtype=np.float64),
        total_rows=10,
        cardinalities=np.asarray([2], dtype=np.int32),
        projection_cfg={"project_partitions": False, "clip_nonpartition": False},
        output_dir=tmp_path / "combined",
        epsilon=1.0,
        measurement_sigma=1.0,
        delta=1.0e-9,
        selection_rho_per_round=0.0,
        base_measurements=base,
    )
    payload = read_json(artifact / "measurements.json")
    adaptive = payload["projection_diagnostics"]["adaptive_selection"]

    assert np.allclose(payload["target_noisy"], [7.6, 2.4, 6.0])
    assert np.allclose(payload["variances"], [0.8, 0.8, 4.0])
    assert np.isclose(payload["rho_spent"], 1.0)
    assert adaptive["coverage_complete"] is True
    assert np.isclose(adaptive["coverage_rho"], 0.5)
    assert np.isclose(adaptive["adaptive_rho_spent"], 0.5)
    assert len(payload["groups"]) == 2


def test_adaptive_measurements_precision_combine_variable_sigmas(tmp_path: Path) -> None:
    module = _load_script("run_adaptive_selection_ablation")
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "x=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "x=1", "oneway:0", "oneway")
    qcat = builder.build()
    schema = TableSchema(
        columns=[ColumnSchema(name="x", kind="categorical", cardinality=2, categories=["0", "1"])]
    )
    group = MeasurementGroup(
        query_indices=np.asarray([0, 1], dtype=np.int32),
        sensitivity_l2=1.0,
        rho=0.125,
        sigma=2.0,
        noise_std=2.0,
        name="oneway:0",
        family="oneway",
        is_partition=True,
    )
    base = Measurements(
        target_noisy=np.asarray([6.0, 4.0], dtype=np.float32),
        target_projected=np.asarray([6.0, 4.0], dtype=np.float32),
        variances=np.asarray([4.0, 4.0], dtype=np.float32),
        inv_variances=np.asarray([0.25, 0.25], dtype=np.float32),
        groups=[group],
        mode="dp",
        rho_total=0.125,
        rho_spent=0.125,
        epsilon_delta=1.0,
        delta=1.0e-9,
        num_rows=10,
    )
    block = module.AdaptiveBlock(
        name="oneway:0",
        family="oneway",
        query_indices=np.asarray([0, 1], dtype=np.int32),
        delta_l2=1.0,
        is_vector=True,
        scope=(0,),
        coverage_weight=1.0,
    )
    records = [
        module.MeasurementRecord(
            block_id=0,
            noisy=np.asarray([8.0, 2.0], dtype=np.float64),
            measurement_sigma=1.0,
        ),
        module.MeasurementRecord(
            block_id=0,
            noisy=np.asarray([4.0, 6.0], dtype=np.float64),
            measurement_sigma=2.0,
        ),
    ]

    artifact = module._measurement_artifact(
        qcat=qcat,
        schema=schema,
        blocks=[block],
        measurement_records=records,
        current_syn_answers=np.asarray([5.0, 5.0], dtype=np.float64),
        total_rows=10,
        cardinalities=np.asarray([2], dtype=np.int32),
        projection_cfg={"project_partitions": False, "clip_nonpartition": False},
        output_dir=tmp_path / "variable_sigma",
        epsilon=1.0,
        measurement_sigma=9.0,
        delta=1.0e-9,
        selection_rho_per_round=0.0,
        base_measurements=base,
    )
    payload = read_json(artifact / "measurements.json")
    adaptive = payload["projection_diagnostics"]["adaptive_selection"]

    assert np.allclose(payload["target_noisy"], [7.0, 3.0])
    assert np.allclose(payload["variances"], [2.0 / 3.0, 2.0 / 3.0])
    assert np.isclose(adaptive["adaptive_measurement_rho"], 0.625)
    assert adaptive["measurement_sigma_schedule"] == [1.0, 2.0]
    assert np.isclose(payload["rho_spent"], 0.75)


def _manifest(tmp_path: Path, audit_module, *, mode: str = "smoke"):
    artifact_names = sorted(audit_module.REQUIRED_ARTIFACTS)
    artifacts = {}
    for name in artifact_names:
        path = tmp_path / f"{name}.artifact"
        path.write_text(f"artifact:{name}\n", encoding="utf-8")
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": audit_module.sha256_file(path),
        }
    rho = 0.01
    delta = 1.0e-9
    epsilon = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    final = mode == "final"
    return {
        "protocol_id": audit_module.PROTOCOL_ID,
        "protocol_mode": mode,
        "paper_evidence_candidate": False,
        "paper_evidence_qualified": False,
        "privacy": {
            "mode": "dp",
            "adjacency": "add_remove_one",
            "epsilon": epsilon,
            "delta": delta,
            "rho_total": rho,
            "epsilon_recomputed": epsilon,
            "selection_rho_per_round": 0.0005 / (50 if final else 2),
            "measurement_rho_per_round": 0.0045 / (50 if final else 2),
            "coverage_rho": 0.005,
        },
        "method": {
            "label": "SAGE-QDTE",
            "scheme": "voi_sageordergain_harmonic_qproject_repeat",
            "selection_input": "private_true_answers",
            "selection_mechanism": "exponential",
            "selection_score_sensitivity": 1.0,
            "selection_ledger": "bounded_range_em_zcdp",
            "selection_rule": "sample",
            "public_bootstrap_rounds": 0,
            "rounds": 50 if final else 2,
            "initial_fit_iters": 5000 if final else 1,
            "inner_iters": 5000 if final else 1,
            "final_refit_iters": 5000 if final else 1,
            "generator_profile": "qdte_standard",
            "initialization": "dp_oneway_independent",
            "coverage_mode": "oneway_only",
            "generator_seed_mode": "per_stage",
            "workload": "complete_orthogonal_oneway_twoway",
            "projection_profile": "P1_lightweight",
            "coverage_rho_fraction": 0.1,
            "adaptive_rho_fraction": 0.9,
            "adaptive_measurement_fraction": 0.9,
        },
        "checks": {"privacy": True, "hashes": True},
        "artifacts": artifacts,
        "source_tree_sha256": "0" * 64,
    }


def test_integrated_manifest_audit_verifies_artifact_hashes(tmp_path: Path) -> None:
    module = _load_script("audit_integrated_sage_qdte_run")
    manifest = _manifest(tmp_path, module)

    assert module.audit_manifest(manifest) == []

    artifact = Path(manifest["artifacts"]["schema"]["path"])
    artifact.write_text("tampered\n", encoding="utf-8")
    errors = module.audit_manifest(manifest)

    assert any("schema" in error and "changed" in error for error in errors)


def test_final_manifest_requires_full_refit_and_qualification(tmp_path: Path) -> None:
    module = _load_script("audit_integrated_sage_qdte_run")
    manifest = _manifest(tmp_path, module, mode="final")

    assert module.audit_manifest(manifest) == []

    manifest["method"]["final_refit_iters"] = 20
    manifest["paper_evidence_qualified"] = True
    errors = module.audit_manifest(manifest, verify_files=False)

    assert any("50/5000/5000" in error for error in errors)
    assert any("cannot be paper_evidence_qualified" in error for error in errors)


def test_production_runner_hardcodes_certified_private_em_call() -> None:
    source = (ROOT / "scripts" / "run_integrated_sage_qdte.py").read_text(encoding="utf-8")

    assert 'selection_input="oracle"' in source
    assert 'selection_ledger="conservative"' in source
    assert 'selection_rule="sample"' in source
    assert "_certified_private_selection_sensitivity" not in source
    assert 'generator_profile="qdte_standard"' in source
    assert 'generator_seed_mode="per_stage"' in source
    assert "initial_fit_iters=FINAL_INITIAL_FIT_ITERS" in source
    assert "true_answers=true_answers" in source
    assert "promotion_true_utility" not in source
