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


def _valid_dp_release_config() -> dict:
    cfg = _valid_config()
    cfg["privacy"].update(
        {
            "mode": "dp",
            "dp_release_mode": True,
            "public_row_count": True,
            "public_n_rows": 4,
            "adjacency": "add_remove",
        }
    )
    cfg["preprocess"] = {"public_schema_json": "sibling"}
    cfg["evaluation"].update(
        {
            "compute_true_query_error": False,
            "compute_heldout_query_error": False,
        }
    )
    return cfg


def test_dp_release_config_passes_with_explicit_public_inputs() -> None:
    validate_config(_valid_dp_release_config())


def test_transcript_only_generation_config_passes_without_private_input() -> None:
    cfg = _valid_dp_release_config()
    cfg["run"] = {"transcript_only_generation": True}
    cfg["measurement"] = {"reuse_from": "released_transcript"}
    cfg["workload"]["reuse_from_measurement"] = True

    validate_config(cfg)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["run"].update(input_csv="private.csv"), "forbids run.input_csv"),
        (lambda cfg: cfg.pop("measurement"), "requires measurement.reuse_from"),
        (
            lambda cfg: cfg["workload"].update(reuse_from_measurement=False),
            "reuse_from_measurement",
        ),
        (
            lambda cfg: cfg["init"].update(encoded_npy="real_encoded.npy"),
            "forbids init.encoded_npy",
        ),
    ],
)
def test_transcript_only_generation_config_fails_closed(mutation, message: str) -> None:
    cfg = _valid_dp_release_config()
    cfg["run"] = {"transcript_only_generation": True}
    cfg["measurement"] = {"reuse_from": "released_transcript"}
    cfg["workload"]["reuse_from_measurement"] = True
    mutation(cfg)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["preprocess"].pop("public_schema_json"), "public_schema_json"),
        (lambda cfg: cfg["privacy"].update(public_row_count=False), "public_row_count"),
        (lambda cfg: cfg["privacy"].pop("public_n_rows"), "public_n_rows"),
        (lambda cfg: cfg["privacy"].update(public_n_rows=0), "public_n_rows"),
        (lambda cfg: cfg["privacy"].update(adjacency="replace_one"), "adjacency"),
        (
            lambda cfg: cfg["evaluation"].update(compute_true_query_error=True),
            "in-process private evaluation",
        ),
    ],
)
def test_dp_release_config_fails_closed(mutation, message: str) -> None:
    cfg = _valid_dp_release_config()
    mutation(cfg)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


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
        ("qdte", "transport_prefix_strategy", "unknown", ValueError),
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


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("privacy", "rho_total", 0.0),
        ("run", "seed", -1),
        ("privacy", "delta", 1.0),
        ("privacy", "min_variance", 0.0),
        ("workload", "max_queries", 0),
        ("workload", "max_terms", 0),
        ("workload", "random_seed", -1),
        ("preprocess", "numerical_bins", 0),
        ("preprocess", "auto_numeric_min_unique", 0),
        ("runtime", "answer_batch_size", 0),
        ("qdte", "num_active_targets", 0),
        ("qdte", "total_candidates_per_iter", 0),
        ("qdte", "stop_patience", 0),
        ("qdte", "kappa_noise", -1.0),
        ("qdte", "lambda_cost", -1.0),
        ("qdte", "numerical_distance_gamma", -1.0),
        ("qdte", "random_candidate_fraction", 1.1),
        ("qdte", "min_advantage", -1.0e-6),
        ("qdte", "gpu_source_draws", 0),
        ("qdte", "gpu_return_oversample_factor", 0),
        ("qdte", "gpu_sparse_changed_attr_capacity", 0),
        ("qdte", "debt_decay", 1.1),
        ("qdte", "debt_repay", -1.0),
        ("qdte", "debt_cap", 0.0),
        ("debug", "loss_tolerance", -1.0),
    ],
)
def test_invalid_core_numeric_controls_fail_fast(section: str, key: str, value: object) -> None:
    cfg = _valid_config()
    cfg.setdefault(section, {})[key] = value

    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_legacy_eval_every_fails_instead_of_being_silently_ignored() -> None:
    cfg = _valid_config()
    cfg["qdte"]["eval_every"] = 100

    with pytest.raises(ValueError, match="eval_every"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("minimum_key", "maximum_key"),
    [
        ("random_group_min_size", "random_group_max_size"),
        ("directed_group_min_size", "directed_group_max_size"),
        ("directed_group_augment_min_size", "directed_group_augment_max_size"),
    ],
)
def test_group_size_ranges_fail_fast(minimum_key: str, maximum_key: str) -> None:
    cfg = _valid_config()
    cfg["qdte"][minimum_key] = 4
    cfg["qdte"][maximum_key] = 2

    with pytest.raises(ValueError, match=maximum_key):
        validate_config(cfg)


def test_nonpositive_synthetic_row_count_fails_fast() -> None:
    cfg = _valid_config()
    cfg["init"]["N_syn"] = 0

    with pytest.raises(ValueError, match="init.N_syn"):
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
            "certificate_feasibility_tolerance": 1.0e-6,
            "certificate_gap_absolute_tolerance": 1.0e-7,
            "certificate_gap_relative_tolerance": 1.0e-8,
            "certificate_max_iterations": 100,
        }
    }

    validate_config(cfg)


@pytest.mark.parametrize(
    "key",
    [
        "certificate_feasibility_tolerance",
        "certificate_gap_absolute_tolerance",
        "certificate_gap_relative_tolerance",
    ],
)
def test_query_space_feasible_lsq_rejects_negative_certificate_tolerance(key: str) -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "consistency": {
            "enabled": True,
            "method": "query_space_feasible_lsq",
            key: -1.0,
        }
    }

    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_query_space_feasible_lsq_rejects_nonpositive_certificate_iterations() -> None:
    cfg = _valid_config()
    cfg["projection"] = {
        "consistency": {
            "enabled": True,
            "method": "query_space_feasible_lsq",
            "certificate_max_iterations": 0,
        }
    }

    with pytest.raises(ValueError, match="certificate_max_iterations"):
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


def test_measurement_fission_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "train_fraction": 0.8,
            "checkpoint_interval": 100,
            "selection_rule": "earliest_within_one_se",
            "one_se_multiplier": 1.0,
            "seed_offset": 51_771,
        }
    }

    validate_config(cfg)


def test_measurement_fission_validation_minimum_is_allowed() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "selection_rule": "validation_minimum",
        }
    }

    validate_config(cfg)


def test_measurement_fission_reverse_optimization_branch_is_allowed() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "train_fraction": 0.5,
            "optimization_branch": "validation",
        }
    }

    validate_config(cfg)


def test_measurement_fission_rejects_unknown_optimization_branch() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "optimization_branch": "both",
        }
    }

    with pytest.raises(ValueError, match="optimization_branch"):
        validate_config(cfg)


def test_measurement_fission_l2_tvd_selection_is_allowed() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "selection_rule": "l2_one_se_tvd_minimum",
        }
    }

    validate_config(cfg)


def test_measurement_fission_l2_tvd_upper_selection_is_allowed() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "selection_rule": "l2_one_se_tvd_upper_minimum",
        }
    }

    validate_config(cfg)


@pytest.mark.parametrize("train_fraction", [0.0, 1.0])
def test_measurement_fission_rejects_invalid_train_fraction(train_fraction: float) -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "train_fraction": train_fraction,
        }
    }

    with pytest.raises(ValueError, match="train_fraction"):
        validate_config(cfg)


def test_measurement_fission_rejects_projection_uncertainty() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {"fission": {"enabled": True}}
    cfg["projection"] = {
        "uncertainty": {
            "enabled": True,
            "method": "bootstrap_diagonal",
            "num_samples": 16,
        }
    }

    with pytest.raises(ValueError, match="projection.uncertainty.enabled=false"):
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


def test_structured_swap_controls_are_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "structured_swap_enabled": True,
            "structured_swap_start_iter": 1,
            "structured_swap_interval": 10,
            "structured_swap_candidate_units": 2048,
            "structured_swap_transport_pool": 256,
            "structured_swap_accept_start": 16,
            "structured_swap_accept_end": 1,
            "structured_swap_accept_schedule": "cosine",
            "structured_swap_noise_guard_kappa": 1.0,
            "structured_swap_noise_guard_mode": "bonferroni",
            "structured_swap_noise_guard_alpha": 0.05,
            "structured_swap_trigger_rms": 1.0,
            "structured_swap_delta_backend": "dense_gpu",
            "structured_swap_exploration_floor": 0.2,
            "structured_swap_policy_decay": 0.95,
            "structured_swap_policy_prior_strength": 32.0,
        }
    )

    validate_config(cfg)


def test_structured_swap_rejects_unknown_delta_backend() -> None:
    cfg = _valid_config()
    cfg["qdte"]["structured_swap_delta_backend"] = "unknown"

    with pytest.raises(ValueError, match="structured_swap_delta_backend"):
        validate_config(cfg)


def test_structured_swap_rejects_invalid_noise_guard_controls() -> None:
    cfg = _valid_config()
    cfg["qdte"]["structured_swap_noise_guard_mode"] = "unknown"

    with pytest.raises(ValueError, match="structured_swap_noise_guard_mode"):
        validate_config(cfg)

    cfg = _valid_config()
    cfg["qdte"]["structured_swap_noise_guard_alpha"] = 1.0

    with pytest.raises(ValueError, match="structured_swap_noise_guard_alpha"):
        validate_config(cfg)


def test_structured_swap_allows_tvd_l1_with_deterministic_guard() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "structured_swap_enabled": True,
            "objective_loss": "tvd_l1",
            "candidate_backend": "cpu_repair",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_noise_guard_kappa": 0.0,
            "structured_swap_noise_guard_mode": "fixed",
        }
    )

    validate_config(cfg)


def test_structured_swap_tvd_l1_rejects_quadratic_noise_guard() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "structured_swap_enabled": True,
            "objective_loss": "tvd_l1",
            "candidate_backend": "cpu_repair",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_noise_guard_kappa": 1.0,
        }
    )

    with pytest.raises(ValueError, match="noise_guard_kappa=0"):
        validate_config(cfg)


def test_tvd_l1_objective_loss_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "objective_loss": "tvd_l1",
            "candidate_backend": "cpu_repair",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
        }
    )

    validate_config(cfg)


def test_fission_allows_unweighted_tvd_l1_training() -> None:
    cfg = _valid_config()
    cfg["measurement"] = {
        "fission": {
            "enabled": True,
            "train_fraction": 0.8,
            "checkpoint_interval": 10,
            "selection_rule": "l2_one_se_tvd_minimum",
        }
    }
    cfg["qdte"].update(
        {
            "objective_loss": "tvd_l1",
            "objective_weighting": "unweighted",
            "candidate_backend": "cpu_repair",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
        }
    )

    validate_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_backend": "jax_repair"}, "candidate_backend"),
        ({"transport_mode": "microbatch_greedy"}, "transport_mode"),
        ({"atom_flow_update_mode": "exact"}, "atom_flow_update_mode"),
    ],
)
def test_tvd_l1_rejects_incomplete_objective_paths(overrides: dict[str, object], message: str) -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "objective_loss": "tvd_l1",
            "candidate_backend": "cpu_repair",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            **overrides,
        }
    )

    with pytest.raises(ValueError, match=message):
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


@pytest.mark.parametrize("score_backend", ["target_only", "sparse_delta"])
def test_gpu_candidate_backend_rejects_nonfused_score_backends(score_backend: str) -> None:
    cfg = _valid_config()
    cfg["qdte"]["candidate_backend"] = "jax_repair"
    cfg["qdte"]["score_backend"] = score_backend

    with pytest.raises(ValueError, match="fused scoring"):
        validate_config(cfg)


def test_orthogonal_precision_reference_path_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "precision_operator": "orthogonal_interaction",
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_enabled": False,
        }
    )

    validate_config(cfg)


@pytest.mark.parametrize(
    "precision_operator",
    [
        "orthogonal_interaction_shrink_raw",
        "orthogonal_interaction_shrink_analytic",
        "orthogonal_interaction_p3_raw",
        "orthogonal_interaction_p3_bootdiag",
        "orthogonal_interaction_p3_active_set",
    ],
)
def test_extended_orthogonal_precision_reference_paths_are_allowed(
    precision_operator: str,
) -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "precision_operator": precision_operator,
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_enabled": False,
        }
    )

    validate_config(cfg)


def test_orthogonal_precision_chi_square_confidence_stop_is_allowed() -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "precision_operator": "orthogonal_interaction",
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_enabled": False,
            "confidence_stop": {
                "enabled": True,
                "method": "chi_square",
                "alpha": 0.05,
            },
        }
    )

    validate_config(cfg)


def _entropy_config() -> dict:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "precision_operator": "orthogonal_interaction",
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_enabled": False,
            "allow_below_noise_fallback": True,
            "entropy": {
                "enabled": True,
                "method": "confidence_constrained_product_kl_primal_dual_v1",
                "alpha": 0.05,
                "product_prior_smoothing": 1.0,
                "dual_initial": 1.0,
            },
        }
    )
    return cfg


def test_confidence_constrained_entropy_reference_path_is_allowed() -> None:
    validate_config(_entropy_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["qdte"].update(precision_operator="diagonal"),
            "orthogonal_interaction",
        ),
        (
            lambda cfg: cfg["qdte"].update(allow_below_noise_fallback=False),
            "allow_below_noise_fallback",
        ),
        (
            lambda cfg: cfg["qdte"].update(structured_swap_enabled=True),
            "structured",
        ),
        (
            lambda cfg: cfg["qdte"].update(confidence_stop={"enabled": True}),
            "Disable qdte.confidence_stop",
        ),
        (
            lambda cfg: cfg["qdte"]["entropy"].update(alpha=0.1),
            "alpha=0.05",
        ),
        (
            lambda cfg: cfg["qdte"]["entropy"].update(product_prior_smoothing=0.5),
            "product_prior_smoothing=1.0",
        ),
        (
            lambda cfg: cfg["qdte"]["entropy"].update(dual_initial=2.0),
            "dual_initial=1.0",
        ),
    ],
)
def test_confidence_constrained_entropy_fails_closed(mutation, message: str) -> None:
    cfg = _entropy_config()
    mutation(cfg)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_chi_square_confidence_stop_rejects_diagonal_precision() -> None:
    cfg = _valid_config()
    cfg["qdte"]["confidence_stop"] = {"enabled": True, "alpha": 0.05}

    with pytest.raises(ValueError, match="orthogonal_interaction"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("confidence_stop", "message"),
    [
        ([], "must be a mapping"),
        ({"enabled": "true"}, "enabled must be boolean"),
        ({"alpha": 0.0}, "alpha must be in"),
        ({"alpha": 1.0}, "alpha must be in"),
        ({"method": "normal"}, "method must be one of"),
    ],
)
def test_chi_square_confidence_stop_rejects_invalid_config(
    confidence_stop: object,
    message: str,
) -> None:
    cfg = _valid_config()
    cfg["qdte"]["confidence_stop"] = confidence_stop

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"score_backend": "dense_gpu"}, "score_backend"),
        ({"candidate_backend": "jax_repair"}, "GPU candidate backends"),
        ({"transport_mode": "microbatch_greedy"}, "batch atom_flow"),
        ({"structured_swap_enabled": True}, "structured_swap"),
        ({"objective_weight_profile": "joint_utility_envelope"}, "no query weight profile"),
    ],
)
def test_orthogonal_precision_rejects_unimplemented_combinations(
    overrides: dict[str, object],
    message: str,
) -> None:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "precision_operator": "orthogonal_interaction",
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_enabled": False,
            **overrides,
        }
    )

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def _interaction_cycle_config() -> dict:
    cfg = _valid_config()
    cfg["qdte"].update(
        {
            "candidate_backend": "cpu_repair",
            "score_backend": "precision_operator",
            "precision_operator": "orthogonal_interaction",
            "objective_loss": "quadratic",
            "objective_weighting": "variance",
            "objective_weight_profile": "none",
            "transport_mode": "atom_flow",
            "atom_flow_update_mode": "batch",
            "structured_swap_enabled": True,
            "structured_swap_compiler": "interaction_rectangle_v1",
            "structured_swap_delta_backend": "sparse_cpu",
        }
    )
    return cfg


def test_exact_interaction_cycle_reference_path_is_allowed() -> None:
    validate_config(_interaction_cycle_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["qdte"].update(precision_operator="diagonal"),
            "exact raw",
        ),
        (
            lambda cfg: cfg["qdte"].update(
                precision_operator="orthogonal_interaction_shrink_raw"
            ),
            "exact raw",
        ),
        (
            lambda cfg: cfg["qdte"].update(structured_swap_delta_backend="dense_gpu"),
            "sparse_cpu",
        ),
        (
            lambda cfg: cfg["qdte"].update(confidence_stop={"enabled": True}),
            "does not support confidence stopping",
        ),
    ],
)
def test_exact_interaction_cycle_fails_closed(mutation, message: str) -> None:
    cfg = _interaction_cycle_config()
    mutation(cfg)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)
