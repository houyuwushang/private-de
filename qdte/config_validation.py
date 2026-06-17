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

    measurement_mode = str(privacy.get("measurement_mode", "static_all")).lower()
    if measurement_mode != "static_all":
        raise NotImplementedError(
            "Only privacy.measurement_mode=static_all is implemented. "
            "adaptive_select_measure/select-measure-generate is not implemented in this version."
        )

    heldout_workload = evaluation.get("heldout_workload", {})
    if heldout_workload is not None:
        if not isinstance(heldout_workload, dict):
            raise ValueError("evaluation.heldout_workload must be a mapping")

    if bool(evaluation.get("downstream_ml", False)):
        raise NotImplementedError("evaluation.downstream_ml=true is not implemented")

    candidate_backend = str(qdte.get("candidate_backend", "cpu_repair"))
    score_backend = str(qdte.get("score_backend", "dense_gpu"))
    candidate_compiler = str(qdte.get("candidate_compiler", "single_query"))
    if bool(workload.get("include_halfspace", False)) and candidate_backend in {"jax_repair", "gpu_repair"}:
        raise NotImplementedError(
            "workload.include_halfspace=true currently requires qdte.candidate_backend=cpu_repair; "
            "GPU fused candidate repair for halfspace is not implemented"
        )
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
