from __future__ import annotations

from typing import Any


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_choice(section: dict[str, Any], key: str, allowed: set[str], default: str, dotted: str) -> None:
    value = section.get(key, default)
    if value is None:
        value = default
    normalized = str(value)
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{dotted} must be one of: {allowed_text}; got {normalized!r}")


def validate_config(config: dict[str, Any]) -> None:
    privacy = _section(config, "privacy")
    workload = _section(config, "workload")
    evaluation = _section(config, "evaluation")
    init = _section(config, "init")
    qdte = _section(config, "qdte")
    projection = _section(config, "projection")
    measurement = _section(config, "measurement")
    population = _section(config, "population")

    measurement_mode = str(privacy.get("measurement_mode", "static_all")).lower()
    if measurement_mode != "static_all":
        raise NotImplementedError(
            "Only privacy.measurement_mode=static_all is implemented. "
            "adaptive_select_measure/select-measure-generate is not implemented in this version."
        )
    reuse_from = measurement.get("reuse_from", measurement.get("artifact_dir"))
    if reuse_from is not None and str(reuse_from) == "":
        raise ValueError("measurement.reuse_from must be a non-empty path when provided")

    heldout_workload = evaluation.get("heldout_workload", {})
    if heldout_workload is not None:
        if not isinstance(heldout_workload, dict):
            raise ValueError("evaluation.heldout_workload must be a mapping")
    oracle_bias = evaluation.get("oracle_projection_bias", {})
    if oracle_bias is not None:
        if not isinstance(oracle_bias, dict):
            raise ValueError("evaluation.oracle_projection_bias must be a mapping")
        if bool(oracle_bias.get("enabled", False)):
            if int(oracle_bias.get("num_samples", 32)) <= 1:
                raise ValueError("evaluation.oracle_projection_bias.num_samples must be greater than 1")

    if bool(evaluation.get("downstream_ml", False)):
        raise NotImplementedError("evaluation.downstream_ml=true is not implemented")

    candidate_backend = str(qdte.get("candidate_backend", "cpu_repair"))
    score_backend = str(qdte.get("score_backend", "dense_gpu"))
    candidate_compiler = str(qdte.get("candidate_compiler", "single_query"))
    if score_backend == "sparse_delta_gpu" and candidate_backend not in {"jax_repair", "gpu_repair"}:
        raise ValueError("qdte.score_backend='sparse_delta_gpu' requires qdte.candidate_backend to be jax_repair or gpu_repair")
    if candidate_compiler != "single_query" and candidate_backend in {"jax_repair", "gpu_repair"}:
        raise NotImplementedError(
            "qdte.candidate_compiler other than 'single_query' currently requires qdte.candidate_backend=cpu_repair"
        )

    consistency = projection.get("consistency", {})
    if consistency is not None:
        if not isinstance(consistency, dict):
            raise ValueError("projection.consistency must be a mapping")
        if bool(consistency.get("enabled", False)):
            method = str(consistency.get("method", "local_marginal_ipf"))
            if method not in {
                "local_marginal_ipf",
                "query_space_lsq",
                "query_space_feasible_lsq",
                "local_table_feasible_lsq",
                "local_table_feasible_jax",
            }:
                raise ValueError(
                    "projection.consistency.method must be one of: "
                    "local_marginal_ipf, query_space_lsq, query_space_feasible_lsq, "
                    "local_table_feasible_lsq, local_table_feasible_jax"
                )
            if method in {"local_marginal_ipf", "local_table_feasible_lsq", "local_table_feasible_jax"}:
                if int(consistency.get("max_scope_cells", 200_000)) <= 0:
                    raise ValueError("projection.consistency.max_scope_cells must be positive")
            if method == "local_marginal_ipf":
                if int(consistency.get("max_iterations", 100)) <= 0:
                    raise ValueError("projection.consistency.max_iterations must be positive")
                if int(consistency.get("max_lsq_iterations", 100)) <= 0:
                    raise ValueError("projection.consistency.max_lsq_iterations must be positive")
            if method in {"query_space_lsq", "query_space_feasible_lsq"}:
                if int(consistency.get("max_constraints", 200_000)) <= 0:
                    raise ValueError("projection.consistency.max_constraints must be positive")
                if float(consistency.get("solver_atol", 1.0e-10)) < 0.0:
                    raise ValueError("projection.consistency.solver_atol must be non-negative")
                if float(consistency.get("solver_btol", 1.0e-10)) < 0.0:
                    raise ValueError("projection.consistency.solver_btol must be non-negative")
            if method == "query_space_lsq":
                if int(consistency.get("solver_max_iterations", 10_000)) <= 0:
                    raise ValueError("projection.consistency.solver_max_iterations must be positive")
            if method in {"query_space_feasible_lsq", "local_table_feasible_lsq"}:
                if int(consistency.get("solver_max_iterations", 1_000)) <= 0:
                    raise ValueError("projection.consistency.solver_max_iterations must be positive")
                if float(consistency.get("solver_ftol", 1.0e-9)) < 0.0:
                    raise ValueError("projection.consistency.solver_ftol must be non-negative")
                if int(consistency.get("max_dense_constraint_cells", 20_000_000)) <= 0:
                    raise ValueError("projection.consistency.max_dense_constraint_cells must be positive")
            if method == "local_table_feasible_jax":
                if int(consistency.get("jax_iterations", 1_000)) <= 0:
                    raise ValueError("projection.consistency.jax_iterations must be positive")
                if float(consistency.get("jax_active_set_tolerance", 1.0e-8)) < 0.0:
                    raise ValueError("projection.consistency.jax_active_set_tolerance must be non-negative")
                if float(consistency.get("jax_kkt_ridge", 1.0e-10)) < 0.0:
                    raise ValueError("projection.consistency.jax_kkt_ridge must be non-negative")
                if int(consistency.get("max_dense_constraint_cells", 20_000_000)) <= 0:
                    raise ValueError("projection.consistency.max_dense_constraint_cells must be positive")
            if float(consistency.get("tolerance", 1.0e-2)) < 0.0:
                raise ValueError("projection.consistency.tolerance must be non-negative")

    uncertainty = projection.get("uncertainty", {})
    if uncertainty is not None:
        if not isinstance(uncertainty, dict):
            raise ValueError("projection.uncertainty must be a mapping")
        if bool(uncertainty.get("enabled", False)):
            method = str(uncertainty.get("method", "bootstrap_diagonal"))
            if method != "bootstrap_diagonal":
                raise ValueError("projection.uncertainty.method must be 'bootstrap_diagonal'")
            if int(uncertainty.get("num_samples", 32)) <= 1:
                raise ValueError("projection.uncertainty.num_samples must be greater than 1")
            center = str(uncertainty.get("center", "projected")).lower()
            if center not in {"projected", "noisy"}:
                raise ValueError("projection.uncertainty.center must be 'projected' or 'noisy'")
            if float(uncertainty.get("min_variance", 1.0e-6)) <= 0.0:
                raise ValueError("projection.uncertainty.min_variance must be positive")
            if float(uncertainty.get("min_raw_variance_fraction", 0.0)) < 0.0:
                raise ValueError("projection.uncertainty.min_raw_variance_fraction must be non-negative")
            debias_alpha = float(uncertainty.get("debias_alpha", 1.0))
            if debias_alpha < 0.0 or debias_alpha > 1.0:
                raise ValueError("projection.uncertainty.debias_alpha must be in [0, 1]")

    init_method = init.get("method")
    if init_method is not None and str(init_method) != "independent_oneway":
        raise ValueError(f"init.method must be 'independent_oneway'; got {init_method!r}")

    _validate_choice(qdte, "candidate_backend", {"cpu_repair", "jax_repair", "gpu_repair"}, "cpu_repair", "qdte.candidate_backend")
    _validate_choice(
        qdte,
        "candidate_compiler",
        {
            "single_query",
            "masked_single_query",
            "masked_single",
            "relaxed_masked_single_query",
            "relaxed_masked_single",
            "paired_query",
            "paired",
            "masked_paired_query",
            "masked_paired",
            "masked_exit_query",
            "masked_exit_protect",
            "directed_exit_only",
            "exit_only",
            "masked_exit_only",
            "masked_directed_exit_only",
            "random_source_directed_exit",
            "random_directed_exit",
            "residual_weighted_mutation",
            "residual_weighted",
            "enumerated_local",
            "local_enumeration",
            "soft_single_query",
            "soft_source_single_query",
            "residual_value_mutation",
            "residual_value",
            "constructive_partner",
            "synthesized_partner",
            "constructive_partner_b2",
            "attached_constructive_partner",
            "constructive_partner_attached",
            "bounded_best_partner",
            "best_partner",
            "exact_best_partner",
            "protected_same_row",
            "protected_repair",
            "constructive_protected",
            "proposal_mixture",
            "mixture_scheduler",
            "qdte_mixture",
            "fair_mixture",
            "directed_mixture",
        },
        "single_query",
        "qdte.candidate_compiler",
    )
    _validate_choice(
        qdte,
        "candidate_shortfall_policy",
        {"random", "none"},
        "random",
        "qdte.candidate_shortfall_policy",
    )
    _validate_choice(
        qdte,
        "score_backend",
        {"dense_gpu", "target_only", "sparse_delta", "sparse_delta_gpu"},
        "dense_gpu",
        "qdte.score_backend",
    )
    _validate_choice(
        qdte,
        "transport_mode",
        {
            "microbatch_greedy",
            "sequential_greedy",
            "atom_flow",
            "blind_accept",
            "constructive_pair",
            "directed_group",
            "random_group",
        },
        "microbatch_greedy",
        "qdte.transport_mode",
    )
    _validate_choice(
        qdte,
        "directed_group_backend",
        {"cpu", "jax", "gpu"},
        "cpu",
        "qdte.directed_group_backend",
    )
    _validate_choice(
        qdte,
        "directed_group_positive_fill_augment_trigger",
        {"always", "loss_gate", "adaptive"},
        "loss_gate",
        "qdte.directed_group_positive_fill_augment_trigger",
    )
    _validate_choice(
        qdte,
        "constructive_pair_group_augment_trigger",
        {"always", "loss_gate", "adaptive"},
        "adaptive",
        "qdte.constructive_pair_group_augment_trigger",
    )
    _validate_choice(
        qdte,
        "objective_weighting",
        {"variance", "unweighted"},
        "variance",
        "qdte.objective_weighting",
    )
    _validate_choice(
        qdte,
        "accepted_per_iter_schedule",
        {"fixed", "none", "linear", "cosine", "exponential"},
        "fixed",
        "qdte.accepted_per_iter_schedule",
    )
    _validate_choice(
        qdte,
        "atom_flow_update_mode",
        {"batch", "exact"},
        "batch",
        "qdte.atom_flow_update_mode",
    )
    _validate_choice(
        qdte,
        "transport_delta_backend",
        {"cpu", "jax_prefix", "gpu_prefix", "sparse_cpu"},
        "cpu",
        "qdte.transport_delta_backend",
    )
    _validate_choice(
        qdte,
        "constructive_partner_delta_backend",
        {"jax", "sparse_cpu", "dense_cpu"},
        "jax",
        "qdte.constructive_partner_delta_backend",
    )
    _validate_choice(
        qdte,
        "best_partner_delta_backend",
        {
            "jax",
            "jax_batch",
            "jax_fixed",
            "batched_jax",
            "gpu_batch",
            "gpu_fixed",
            "sparse_cpu",
            "dense_cpu",
            "dense_unique",
            "unique_dense",
            "cpu_unique",
            "dense_cached",
            "source_cached",
            "dense_source_cached",
            "cpu_cached",
            "cached_cpu",
        },
        "jax",
        "qdte.best_partner_delta_backend",
    )
    _validate_choice(
        qdte,
        "protected_repair_delta_backend",
        {"jax", "sparse_cpu", "dense_cpu"},
        "jax",
        "qdte.protected_repair_delta_backend",
    )
    if "random_candidate_count" in qdte and int(qdte["random_candidate_count"]) < 0:
        raise ValueError("qdte.random_candidate_count must be non-negative")
    if "accepted_per_iter" in qdte and int(qdte["accepted_per_iter"]) <= 0:
        raise ValueError("qdte.accepted_per_iter must be positive")
    if "accepted_per_iter_start" in qdte and int(qdte["accepted_per_iter_start"]) <= 0:
        raise ValueError("qdte.accepted_per_iter_start must be positive")
    if "accepted_per_iter_end" in qdte and int(qdte["accepted_per_iter_end"]) <= 0:
        raise ValueError("qdte.accepted_per_iter_end must be positive")
    if "accepted_per_iter_warmup_iters" in qdte and int(qdte["accepted_per_iter_warmup_iters"]) < 0:
        raise ValueError("qdte.accepted_per_iter_warmup_iters must be non-negative")
    if "accepted_per_iter_anneal_iters" in qdte and int(qdte["accepted_per_iter_anneal_iters"]) <= 0:
        raise ValueError("qdte.accepted_per_iter_anneal_iters must be positive")
    if "directed_candidate_count" in qdte and int(qdte["directed_candidate_count"]) < 0:
        raise ValueError("qdte.directed_candidate_count must be non-negative")
    if "constructive_pair_pool_multiplier" in qdte and int(qdte["constructive_pair_pool_multiplier"]) < 0:
        raise ValueError("qdte.constructive_pair_pool_multiplier must be non-negative")
    if "constructive_pair_max_pool" in qdte and int(qdte["constructive_pair_max_pool"]) < 0:
        raise ValueError("qdte.constructive_pair_max_pool must be non-negative")
    if "constructive_pair_partner_limit" in qdte and int(qdte["constructive_pair_partner_limit"]) < 0:
        raise ValueError("qdte.constructive_pair_partner_limit must be non-negative")
    if "constructive_pair_harm_query_limit" in qdte and int(qdte["constructive_pair_harm_query_limit"]) < 0:
        raise ValueError("qdte.constructive_pair_harm_query_limit must be non-negative")
    if "constructive_pair_max_units" in qdte and int(qdte["constructive_pair_max_units"]) < 0:
        raise ValueError("qdte.constructive_pair_max_units must be non-negative")
    if (
        "constructive_pair_group_augment_threshold" in qdte
        and int(qdte["constructive_pair_group_augment_threshold"]) < 0
    ):
        raise ValueError("qdte.constructive_pair_group_augment_threshold must be non-negative")
    if (
        "constructive_pair_group_augment_max_loss" in qdte
        and float(qdte["constructive_pair_group_augment_max_loss"]) < 0.0
    ):
        raise ValueError("qdte.constructive_pair_group_augment_max_loss must be non-negative")
    if (
        "constructive_pair_group_augment_noise_floor_ratio" in qdte
        and float(qdte["constructive_pair_group_augment_noise_floor_ratio"]) < 0.0
    ):
        raise ValueError("qdte.constructive_pair_group_augment_noise_floor_ratio must be non-negative")
    if (
        "constructive_pair_group_augment_plateau_window" in qdte
        and int(qdte["constructive_pair_group_augment_plateau_window"]) < 0
    ):
        raise ValueError("qdte.constructive_pair_group_augment_plateau_window must be non-negative")
    if (
        "constructive_pair_group_augment_plateau_relative_drop" in qdte
        and float(qdte["constructive_pair_group_augment_plateau_relative_drop"]) < 0.0
    ):
        raise ValueError("qdte.constructive_pair_group_augment_plateau_relative_drop must be non-negative")
    if "random_group_count" in qdte and int(qdte["random_group_count"]) <= 0:
        raise ValueError("qdte.random_group_count must be positive")
    if "random_group_min_size" in qdte and int(qdte["random_group_min_size"]) <= 0:
        raise ValueError("qdte.random_group_min_size must be positive")
    if "random_group_max_size" in qdte and int(qdte["random_group_max_size"]) < 0:
        raise ValueError("qdte.random_group_max_size must be non-negative")
    if "random_group_pool_multiplier" in qdte and int(qdte["random_group_pool_multiplier"]) < 0:
        raise ValueError("qdte.random_group_pool_multiplier must be non-negative")
    if "random_group_max_pool" in qdte and int(qdte["random_group_max_pool"]) < 0:
        raise ValueError("qdte.random_group_max_pool must be non-negative")
    if "directed_group_seed_count" in qdte and int(qdte["directed_group_seed_count"]) < 0:
        raise ValueError("qdte.directed_group_seed_count must be non-negative")
    if "directed_group_min_size" in qdte and int(qdte["directed_group_min_size"]) <= 0:
        raise ValueError("qdte.directed_group_min_size must be positive")
    if "directed_group_max_size" in qdte and int(qdte["directed_group_max_size"]) < 0:
        raise ValueError("qdte.directed_group_max_size must be non-negative")
    if "directed_group_pool_multiplier" in qdte and int(qdte["directed_group_pool_multiplier"]) < 0:
        raise ValueError("qdte.directed_group_pool_multiplier must be non-negative")
    if "directed_group_max_pool" in qdte and int(qdte["directed_group_max_pool"]) < 0:
        raise ValueError("qdte.directed_group_max_pool must be non-negative")
    if (
        "directed_group_positive_fill_augment_threshold" in qdte
        and int(qdte["directed_group_positive_fill_augment_threshold"]) < 0
    ):
        raise ValueError("qdte.directed_group_positive_fill_augment_threshold must be non-negative")
    if (
        "directed_group_positive_fill_augment_max_loss" in qdte
        and float(qdte["directed_group_positive_fill_augment_max_loss"]) < 0.0
    ):
        raise ValueError("qdte.directed_group_positive_fill_augment_max_loss must be non-negative")
    if (
        "directed_group_positive_fill_augment_noise_floor_ratio" in qdte
        and float(qdte["directed_group_positive_fill_augment_noise_floor_ratio"]) < 0.0
    ):
        raise ValueError("qdte.directed_group_positive_fill_augment_noise_floor_ratio must be non-negative")
    if (
        "directed_group_positive_fill_augment_plateau_window" in qdte
        and int(qdte["directed_group_positive_fill_augment_plateau_window"]) < 0
    ):
        raise ValueError("qdte.directed_group_positive_fill_augment_plateau_window must be non-negative")
    if (
        "directed_group_positive_fill_augment_plateau_relative_drop" in qdte
        and float(qdte["directed_group_positive_fill_augment_plateau_relative_drop"]) < 0.0
    ):
        raise ValueError("qdte.directed_group_positive_fill_augment_plateau_relative_drop must be non-negative")
    if "directed_group_augment_seed_count" in qdte and int(qdte["directed_group_augment_seed_count"]) < 0:
        raise ValueError("qdte.directed_group_augment_seed_count must be non-negative")
    if "directed_group_augment_min_size" in qdte and int(qdte["directed_group_augment_min_size"]) <= 0:
        raise ValueError("qdte.directed_group_augment_min_size must be positive")
    if "directed_group_augment_max_size" in qdte and int(qdte["directed_group_augment_max_size"]) < 0:
        raise ValueError("qdte.directed_group_augment_max_size must be non-negative")
    if "directed_group_augment_pool_multiplier" in qdte and int(qdte["directed_group_augment_pool_multiplier"]) < 0:
        raise ValueError("qdte.directed_group_augment_pool_multiplier must be non-negative")
    if "directed_group_augment_max_pool" in qdte and int(qdte["directed_group_augment_max_pool"]) < 0:
        raise ValueError("qdte.directed_group_augment_max_pool must be non-negative")
    if "encoded_npy" in init and str(init["encoded_npy"]) == "":
        raise ValueError("init.encoded_npy must be a non-empty path when provided")
    if "constructive_partner_seed_fraction" in qdte:
        value = float(qdte["constructive_partner_seed_fraction"])
        if value < 0.0 or value > 1.0:
            raise ValueError("qdte.constructive_partner_seed_fraction must be in [0, 1]")
    if "constructive_partner_harm_queries" in qdte and int(qdte["constructive_partner_harm_queries"]) <= 0:
        raise ValueError("qdte.constructive_partner_harm_queries must be positive")
    if "constructive_partner_partners_per_seed" in qdte and int(qdte["constructive_partner_partners_per_seed"]) <= 0:
        raise ValueError("qdte.constructive_partner_partners_per_seed must be positive")
    if "constructive_partner_side_budget" in qdte and int(qdte["constructive_partner_side_budget"]) < 0:
        raise ValueError("qdte.constructive_partner_side_budget must be non-negative")
    if (
        "constructive_partner_source_over_sample_factor" in qdte
        and int(qdte["constructive_partner_source_over_sample_factor"]) <= 0
    ):
        raise ValueError("qdte.constructive_partner_source_over_sample_factor must be positive")
    if "best_partner_seed_fraction" in qdte:
        value = float(qdte["best_partner_seed_fraction"])
        if value < 0.0 or value > 1.0:
            raise ValueError("qdte.best_partner_seed_fraction must be in [0, 1]")
    if "best_partner_harm_queries" in qdte and int(qdte["best_partner_harm_queries"]) <= 0:
        raise ValueError("qdte.best_partner_harm_queries must be positive")
    if "best_partner_partners_per_seed" in qdte and int(qdte["best_partner_partners_per_seed"]) <= 0:
        raise ValueError("qdte.best_partner_partners_per_seed must be positive")
    if "best_partner_source_samples" in qdte and int(qdte["best_partner_source_samples"]) <= 0:
        raise ValueError("qdte.best_partner_source_samples must be positive")
    if "best_partner_repairs_per_source" in qdte and int(qdte["best_partner_repairs_per_source"]) <= 0:
        raise ValueError("qdte.best_partner_repairs_per_source must be positive")
    if "best_partner_side_budget" in qdte and int(qdte["best_partner_side_budget"]) < 0:
        raise ValueError("qdte.best_partner_side_budget must be non-negative")
    if "best_partner_jax_batch_size" in qdte and int(qdte["best_partner_jax_batch_size"]) <= 0:
        raise ValueError("qdte.best_partner_jax_batch_size must be positive")
    if "best_partner_seed_batch_size" in qdte and int(qdte["best_partner_seed_batch_size"]) <= 0:
        raise ValueError("qdte.best_partner_seed_batch_size must be positive")
    if "best_partner_cached_verify_margin" in qdte and float(qdte["best_partner_cached_verify_margin"]) < 0.0:
        raise ValueError("qdte.best_partner_cached_verify_margin must be non-negative")

    if bool(population.get("enabled", False)):
        if int(population.get("size", 4)) <= 0:
            raise ValueError("population.size must be positive")
        if int(population.get("elite_count", 1)) <= 0:
            raise ValueError("population.elite_count must be positive")
        if int(population.get("elite_count", 1)) > int(population.get("size", 4)):
            raise ValueError("population.elite_count must be less than or equal to population.size")
        if int(population.get("generations", 1)) <= 0:
            raise ValueError("population.generations must be positive")
        if int(population.get("inner_iters", qdte.get("max_iters", 100))) < 0:
            raise ValueError("population.inner_iters must be non-negative")
        if int(population.get("seed_stride", 1000)) <= 0:
            raise ValueError("population.seed_stride must be positive")
        crossover = population.get("crossover", {})
        if crossover is not None:
            if not isinstance(crossover, dict):
                raise ValueError("population.crossover must be a mapping")
            mode = str(crossover.get("mode", population.get("crossover_mode", "random_row"))).lower()
            if mode not in {"random_row", "context_aware"}:
                raise ValueError("population.crossover.mode must be one of: random_row, context_aware")
            if int(crossover.get("children", population.get("crossover_children", 0))) < 0:
                raise ValueError("population.crossover.children must be non-negative")
            fraction = float(crossover.get("fraction", population.get("crossover_fraction", 0.5)))
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("population.crossover.fraction must be in [0, 1]")
            if int(crossover.get("parent_pool", population.get("crossover_parent_pool", 0))) < 0:
                raise ValueError("population.crossover.parent_pool must be non-negative")
            if int(crossover.get("candidates", population.get("crossover_candidates", 0))) < 0:
                raise ValueError("population.crossover.candidates must be non-negative")
            if int(crossover.get("max_edits", population.get("crossover_max_edits", 0))) < 0:
                raise ValueError("population.crossover.max_edits must be non-negative")
            if int(crossover.get("inner_iters", population.get("crossover_inner_iters", population.get("inner_iters", 0)))) < 0:
                raise ValueError("population.crossover.inner_iters must be non-negative")
        parallel = population.get("parallel", {})
        if parallel is not None:
            if not isinstance(parallel, dict):
                raise ValueError("population.parallel must be a mapping")
            if int(parallel.get("workers", population.get("parallel_workers", 0))) < 0:
                raise ValueError("population.parallel.workers must be non-negative")
            if int(parallel.get("workers_per_gpu", population.get("parallel_workers_per_gpu", 1))) <= 0:
                raise ValueError("population.parallel.workers_per_gpu must be positive")
            gpu_devices = parallel.get("gpu_devices", population.get("parallel_gpu_devices", "auto"))
            if isinstance(gpu_devices, str) and gpu_devices.strip() == "":
                raise ValueError("population.parallel.gpu_devices must be non-empty")
            if isinstance(gpu_devices, (list, tuple)) and len(gpu_devices) == 0:
                raise ValueError("population.parallel.gpu_devices must be non-empty")
    if "protected_repair_seed_fraction" in qdte:
        value = float(qdte["protected_repair_seed_fraction"])
        if value < 0.0 or value > 1.0:
            raise ValueError("qdte.protected_repair_seed_fraction must be in [0, 1]")
    if "protected_repair_harm_queries" in qdte and int(qdte["protected_repair_harm_queries"]) <= 0:
        raise ValueError("qdte.protected_repair_harm_queries must be positive")
    if "protected_repair_restarts_per_seed" in qdte and int(qdte["protected_repair_restarts_per_seed"]) <= 0:
        raise ValueError("qdte.protected_repair_restarts_per_seed must be positive")
    if (
        "protected_repair_max_protection_passes" in qdte
        and int(qdte["protected_repair_max_protection_passes"]) <= 0
    ):
        raise ValueError("qdte.protected_repair_max_protection_passes must be positive")
