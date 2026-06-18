from __future__ import annotations

import pytest

from qdte.config_validation import validate_config


def _valid_config() -> dict:
    return {
        "privacy": {"measurement_mode": "static_all"},
        "workload": {"include_halfspace": False},
        "evaluation": {"downstream_ml": False},
        "init": {"method": "independent_oneway"},
        "qdte": {
            "candidate_backend": "cpu_repair",
            "score_backend": "dense_gpu",
            "transport_mode": "microbatch_greedy",
            "transport_delta_backend": "cpu",
        },
    }


def test_valid_config_passes() -> None:
    validate_config(_valid_config())


@pytest.mark.parametrize(
    ("section", "key", "value", "exc_type"),
    [
        ("privacy", "measurement_mode", "adaptive_select_measure", NotImplementedError),
        ("evaluation", "downstream_ml", True, NotImplementedError),
        ("init", "method", "something_else", ValueError),
        ("qdte", "candidate_backend", "unknown", ValueError),
        ("qdte", "candidate_compiler", "unknown", ValueError),
        ("qdte", "score_backend", "unknown", ValueError),
        ("qdte", "transport_mode", "unknown", ValueError),
        ("qdte", "directed_group_backend", "unknown", ValueError),
        ("qdte", "accepted_per_iter_schedule", "unknown", ValueError),
        ("qdte", "atom_flow_update_mode", "unknown", ValueError),
        ("qdte", "transport_delta_backend", "unknown", ValueError),
        ("qdte", "constructive_partner_delta_backend", "unknown", ValueError),
        ("qdte", "best_partner_delta_backend", "unknown", ValueError),
        ("qdte", "protected_repair_delta_backend", "unknown", ValueError),
        ("qdte", "objective_weighting", "unknown", ValueError),
    ],
)
def test_unsupported_or_unknown_config_fails_fast(
    section: str, key: str, value: object, exc_type: type[Exception]
) -> None:
    cfg = _valid_config()
    cfg[section][key] = value

    with pytest.raises(exc_type):
        validate_config(cfg)


def test_prefix_monotonicity_is_allowed() -> None:
    cfg = _valid_config()
    cfg["projection"] = {"prefix_monotonicity": True}

    validate_config(cfg)


def test_query_space_lsq_consistency_projection_is_allowed() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "consistency": {
            "enabled": True,
            "method": "query_space_lsq",
            "max_constraints": 100,
            "solver_max_iterations": 100,
        }
    }

    validate_config(cfg)


def test_query_space_feasible_lsq_consistency_projection_is_allowed() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "consistency": {
            "enabled": True,
            "method": "query_space_feasible_lsq",
            "max_constraints": 100,
            "solver_max_iterations": 100,
            "max_dense_constraint_cells": 10000,
        }
    }

    validate_config(cfg)


def test_local_table_feasible_lsq_consistency_projection_is_allowed() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "consistency": {
            "enabled": True,
            "method": "local_table_feasible_lsq",
            "max_scope_cells": 100,
            "solver_max_iterations": 100,
            "max_dense_constraint_cells": 10000,
        }
    }

    validate_config(cfg)


def test_local_table_feasible_jax_consistency_projection_is_allowed() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "consistency": {
            "enabled": True,
            "method": "local_table_feasible_jax",
            "max_scope_cells": 100,
            "jax_iterations": 100,
            "jax_active_set_tolerance": 1.0e-8,
            "jax_kkt_ridge": 1.0e-10,
            "max_dense_constraint_cells": 10000,
        }
    }

    validate_config(cfg)


def test_projection_uncertainty_bootstrap_is_allowed() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "uncertainty": {
            "enabled": True,
            "method": "bootstrap_diagonal",
            "num_samples": 8,
            "center": "projected",
            "debias_target": True,
            "debias_alpha": 0.5,
            "reproject_debiased_target": True,
            "min_variance": 1.0e-6,
            "min_raw_variance_fraction": 0.01,
        }
    }

    validate_config(cfg)


def test_oracle_projection_bias_diagnostic_is_allowed() -> None:
    cfg = _valid_config()
    cfg["evaluation"] = {
        "downstream_ml": False,
        "oracle_projection_bias": {
            "enabled": True,
            "num_samples": 4,
        },
    }

    validate_config(cfg)


def test_projection_uncertainty_invalid_method_fails_fast() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "uncertainty": {
            "enabled": True,
            "method": "unknown",
        }
    }

    with pytest.raises(ValueError, match="projection.uncertainty.method"):
        validate_config(cfg)


def test_projection_uncertainty_invalid_debias_alpha_fails_fast() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "uncertainty": {
            "enabled": True,
            "method": "bootstrap_diagonal",
            "debias_alpha": 1.5,
        }
    }

    with pytest.raises(ValueError, match="projection.uncertainty.debias_alpha"):
        validate_config(cfg)


def test_oracle_projection_bias_invalid_sample_count_fails_fast() -> None:
    cfg = _valid_config()
    cfg["evaluation"] = {
        "downstream_ml": False,
        "oracle_projection_bias": {
            "enabled": True,
            "num_samples": 1,
        },
    }

    with pytest.raises(ValueError, match="evaluation.oracle_projection_bias.num_samples"):
        validate_config(cfg)


def test_unknown_consistency_projection_method_fails_fast() -> None:
    cfg = _valid_config()
    cfg["projection"] = {"consistency": {"enabled": True, "method": "unknown"}}

    with pytest.raises(ValueError, match="projection.consistency.method"):
        validate_config(cfg)


def test_halfspace_workload_is_allowed() -> None:
    cfg = _valid_config()
    cfg["workload"]["include_halfspace"] = True

    validate_config(cfg)


@pytest.mark.parametrize("candidate_backend", ["jax_repair", "gpu_repair"])
def test_halfspace_gpu_candidate_backend_is_allowed(candidate_backend: str) -> None:
    cfg = _valid_config()
    cfg["workload"]["include_halfspace"] = True
    cfg["qdte"]["candidate_backend"] = candidate_backend

    validate_config(cfg)


def test_atom_flow_transport_mode_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["transport_mode"] = "atom_flow"
    cfg["qdte"]["atom_flow_update_mode"] = "batch"

    validate_config(cfg)


def test_blind_accept_transport_mode_is_allowed_for_ablation() -> None:
    cfg = _valid_config()
    cfg["qdte"]["transport_mode"] = "blind_accept"

    validate_config(cfg)


def test_unweighted_objective_weighting_is_allowed_for_ablation() -> None:
    cfg = _valid_config()
    cfg["qdte"]["objective_weighting"] = "unweighted"

    validate_config(cfg)


def test_measurement_reuse_path_is_allowed() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {"reuse_from": "outputs/example_measurement"}

    validate_config(cfg)


def test_population_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["population"] = {
        "enabled": True,
        "size": 4,
        "elite_count": 2,
        "generations": 3,
        "inner_iters": 10,
        "seed_stride": 100,
        "parallel": {
            "enabled": True,
            "gpu_devices": "0,1",
            "workers_per_gpu": 1,
            "workers": 2,
        },
        "crossover": {
            "enabled": True,
            "mode": "context_aware",
            "children": 2,
            "fraction": 0.5,
            "parent_pool": 3,
            "candidates": 128,
            "max_edits": 8,
            "inner_iters": 5,
        },
    }

    validate_config(cfg)


def test_init_encoded_npy_is_allowed() -> None:
    cfg = _valid_config()
    cfg["init"]["encoded_npy"] = "outputs/seed.npy"

    validate_config(cfg)


def test_population_elite_count_must_fit_population_size() -> None:
    cfg = _valid_config()
    cfg["population"] = {"enabled": True, "size": 2, "elite_count": 3}

    with pytest.raises(ValueError, match="population.elite_count"):
        validate_config(cfg)


def test_population_generations_must_be_positive() -> None:
    cfg = _valid_config()
    cfg["population"] = {"enabled": True, "size": 2, "elite_count": 1, "generations": 0}

    with pytest.raises(ValueError, match="population.generations"):
        validate_config(cfg)


def test_population_parallel_workers_per_gpu_must_be_positive() -> None:
    cfg = _valid_config()
    cfg["population"] = {
        "enabled": True,
        "size": 2,
        "elite_count": 1,
        "parallel": {"enabled": True, "workers_per_gpu": 0},
    }

    with pytest.raises(ValueError, match="population.parallel.workers_per_gpu"):
        validate_config(cfg)


def test_sparse_delta_backends_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["score_backend"] = "sparse_delta"
    cfg["qdte"]["transport_delta_backend"] = "sparse_cpu"

    validate_config(cfg)


def test_paired_query_candidate_compiler_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = "paired_query"

    validate_config(cfg)


def test_masked_paired_query_candidate_compiler_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = "masked_paired_query"

    validate_config(cfg)


def test_masked_single_query_candidate_compiler_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = "masked_single_query"

    validate_config(cfg)


def test_relaxed_masked_single_query_candidate_compiler_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = "relaxed_masked_single_query"

    validate_config(cfg)


def test_masked_exit_query_candidate_compiler_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = "masked_exit_query"

    validate_config(cfg)


@pytest.mark.parametrize("compiler", ["directed_exit_only", "masked_exit_only", "random_source_directed_exit"])
def test_exit_ablation_candidate_compilers_are_allowed(compiler: str) -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = compiler

    validate_config(cfg)


@pytest.mark.parametrize(
    "compiler",
    [
        "residual_weighted_mutation",
        "enumerated_local",
        "soft_single_query",
        "residual_value_mutation",
        "constructive_partner",
        "constructive_partner_b2",
        "bounded_best_partner",
        "protected_same_row",
        "proposal_mixture",
        "qdte_mixture",
    ],
)
def test_proposal_ablation_candidate_compilers_are_allowed(compiler: str) -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_compiler"] = compiler

    validate_config(cfg)


def test_candidate_budget_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_compiler": "qdte_mixture",
            "random_candidate_count": 16,
            "directed_candidate_count": 112,
            "candidate_shortfall_policy": "none",
        }
    )

    validate_config(cfg)


def test_constructive_pair_transport_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "transport_mode": "constructive_pair",
            "transport_prefix_strategy": "best_advantage",
            "constructive_pair_pool_multiplier": 8,
            "constructive_pair_max_pool": 128,
            "constructive_pair_partner_limit": 16,
            "constructive_pair_harm_query_limit": 8,
            "constructive_pair_max_units": 64,
            "constructive_pair_min_target_component": 0.0,
        }
    )

    validate_config(cfg)


def test_random_group_transport_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "transport_mode": "random_group",
            "random_group_count": 32,
            "random_group_min_size": 2,
            "random_group_max_size": 8,
            "random_group_pool_multiplier": 0,
            "random_group_max_pool": 256,
        }
    )

    validate_config(cfg)


def test_directed_group_transport_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "transport_mode": "directed_group",
            "directed_group_seed_count": 32,
            "directed_group_min_size": 1,
            "directed_group_max_size": 8,
            "directed_group_pool_multiplier": 0,
            "directed_group_max_pool": 256,
            "directed_group_allow_negative_steps": False,
            "directed_group_backend": "jax",
            "directed_group_positive_fill": True,
            "directed_group_positive_fill_augment": True,
            "directed_group_positive_fill_augment_trigger": "adaptive",
            "directed_group_positive_fill_augment_threshold": 64,
            "directed_group_positive_fill_augment_max_loss": 4000000.0,
            "directed_group_positive_fill_augment_noise_floor_ratio": 2.0,
            "directed_group_positive_fill_augment_plateau_window": 100,
            "directed_group_positive_fill_augment_plateau_relative_drop": 0.01,
            "directed_group_augment_seed_count": 16,
            "directed_group_augment_min_size": 1,
            "directed_group_augment_max_size": 8,
            "directed_group_augment_pool_multiplier": 0,
            "directed_group_augment_max_pool": 512,
        }
    )

    validate_config(cfg)


def test_constructive_pair_group_augment_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "transport_mode": "constructive_pair",
            "constructive_pair_group_augment": True,
            "constructive_pair_group_augment_trigger": "adaptive",
            "constructive_pair_group_augment_threshold": 8,
            "constructive_pair_group_augment_max_loss": 0.0,
            "constructive_pair_group_augment_noise_floor_ratio": 2.0,
            "constructive_pair_group_augment_plateau_window": 100,
            "constructive_pair_group_augment_plateau_relative_drop": 0.01,
            "directed_group_augment_seed_count": 16,
            "directed_group_augment_min_size": 1,
            "directed_group_augment_max_size": 8,
            "directed_group_augment_pool_multiplier": 0,
            "directed_group_augment_max_pool": 256,
        }
    )

    validate_config(cfg)


def test_accepted_per_iter_schedule_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "accepted_per_iter": 8,
            "accepted_per_iter_schedule": "cosine",
            "accepted_per_iter_start": 8,
            "accepted_per_iter_end": 1,
            "accepted_per_iter_warmup_iters": 10,
            "accepted_per_iter_anneal_iters": 100,
        }
    )

    validate_config(cfg)


@pytest.mark.parametrize("backend", ["jax", "sparse_cpu", "dense_cpu"])
def test_constructive_partner_delta_backends_are_allowed(backend: str) -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_compiler": "constructive_partner",
            "constructive_partner_delta_backend": backend,
        }
    )

    validate_config(cfg)


def test_constructive_partner_b2_side_budget_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_compiler": "constructive_partner_b2",
            "constructive_partner_side_budget": 128,
        }
    )

    validate_config(cfg)


def test_bounded_best_partner_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_compiler": "bounded_best_partner",
            "best_partner_seed_fraction": 0.5,
            "best_partner_harm_queries": 4,
            "best_partner_partners_per_seed": 1,
            "best_partner_source_samples": 64,
            "best_partner_repairs_per_source": 2,
            "best_partner_side_budget": 128,
            "best_partner_delta_backend": "jax_batch",
            "best_partner_jax_batch_size": 16384,
            "best_partner_seed_batch_size": 32,
            "best_partner_cached_verify_margin": 1.0e-4,
        }
    )

    validate_config(cfg)


def test_protected_repair_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_compiler": "protected_same_row",
            "protected_repair_seed_fraction": 0.5,
            "protected_repair_harm_queries": 4,
            "protected_repair_restarts_per_seed": 2,
            "protected_repair_max_protection_passes": 2,
            "protected_repair_delta_backend": "dense_cpu",
        }
    )

    validate_config(cfg)


def test_non_default_candidate_compiler_requires_cpu_backend() -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_backend"] = "jax_repair"
    cfg["qdte"]["candidate_compiler"] = "masked_paired_query"

    with pytest.raises(NotImplementedError, match="candidate_compiler"):
        validate_config(cfg)


def test_sparse_delta_gpu_backend_requires_gpu_candidate_backend() -> None:
    cfg = _valid_config()
    cfg["qdte"]["score_backend"] = "sparse_delta_gpu"

    with pytest.raises(ValueError, match="sparse_delta_gpu"):
        validate_config(cfg)

    cfg["qdte"]["candidate_backend"] = "jax_repair"
    validate_config(cfg)
