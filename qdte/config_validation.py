from __future__ import annotations

import math
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


def _finite_float(section: dict[str, Any], key: str, default: float, dotted: str) -> float:
    value = float(section.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{dotted} must be finite")
    return value


def _positive_int(section: dict[str, Any], key: str, default: int, dotted: str, *, allow_zero: bool = False) -> int:
    value = int(section.get(key, default))
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{dotted} must be {qualifier}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    privacy = _section(config, "privacy")
    run = _section(config, "run")
    workload = _section(config, "workload")
    evaluation = _section(config, "evaluation")
    init = _section(config, "init")
    qdte = _section(config, "qdte")
    projection = _section(config, "projection")
    measurement = _section(config, "measurement")
    population = _section(config, "population")
    preprocess = _section(config, "preprocess")
    runtime = _section(config, "runtime")
    debug = _section(config, "debug")

    _positive_int(run, "seed", 0, "run.seed", allow_zero=True)
    _positive_int(workload, "random_seed", 0, "workload.random_seed", allow_zero=True)

    _validate_choice(privacy, "mode", {"dp", "oracle"}, "dp", "privacy.mode")
    privacy_mode = str(privacy.get("mode", "dp")).lower()
    _validate_choice(
        privacy,
        "adjacency",
        {"add_remove"},
        "add_remove",
        "privacy.adjacency",
    )
    dp_release_mode = bool(privacy.get("dp_release_mode", False))
    if dp_release_mode and privacy_mode != "dp":
        raise ValueError("privacy.dp_release_mode=true requires privacy.mode='dp'")
    rho_total = _finite_float(privacy, "rho_total", 1.0, "privacy.rho_total")
    if privacy_mode == "dp" and rho_total <= 0.0:
        raise ValueError("privacy.rho_total must be positive in DP mode")
    if privacy_mode == "oracle" and rho_total < 0.0:
        raise ValueError("privacy.rho_total must be non-negative")
    delta = _finite_float(privacy, "delta", 1.0e-9, "privacy.delta")
    if not 0.0 < delta < 1.0:
        raise ValueError("privacy.delta must be in (0, 1)")
    if _finite_float(privacy, "min_variance", 1.0e-6, "privacy.min_variance") <= 0.0:
        raise ValueError("privacy.min_variance must be positive")
    if _finite_float(privacy, "oracle_variance", 1.0, "privacy.oracle_variance") <= 0.0:
        raise ValueError("privacy.oracle_variance must be positive")
    allocation = privacy.get("measurement_allocation", {})
    if allocation is not None and not isinstance(allocation, dict):
        raise ValueError("privacy.measurement_allocation must be a mapping")
    public_schema_json = preprocess.get("public_schema_json")
    if public_schema_json is not None and not str(public_schema_json).strip():
        raise ValueError("preprocess.public_schema_json must be a non-empty path")
    if dp_release_mode:
        if public_schema_json is None:
            raise ValueError("privacy.dp_release_mode=true requires preprocess.public_schema_json")
        if privacy.get("public_row_count") is not True:
            raise ValueError("privacy.dp_release_mode=true requires privacy.public_row_count=true")
        public_n_rows = privacy.get("public_n_rows")
        if (
            not isinstance(public_n_rows, int)
            or isinstance(public_n_rows, bool)
            or public_n_rows <= 0
        ):
            raise ValueError(
                "privacy.dp_release_mode=true requires a positive integer "
                "privacy.public_n_rows"
            )
        if str(privacy.get("adjacency", "")) != "add_remove":
            raise ValueError("privacy.dp_release_mode=true requires privacy.adjacency='add_remove'")

    _positive_int(workload, "max_queries", 10_000, "workload.max_queries")
    _positive_int(workload, "max_terms", 4, "workload.max_terms")
    _positive_int(workload, "max_2way_cells", 5_000, "workload.max_2way_cells", allow_zero=True)
    _positive_int(
        workload,
        "exact_group_sensitivity_max_cells",
        200_000,
        "workload.exact_group_sensitivity_max_cells",
    )
    _positive_int(preprocess, "numerical_bins", 32, "preprocess.numerical_bins")
    _positive_int(
        preprocess,
        "auto_numeric_min_unique",
        10,
        "preprocess.auto_numeric_min_unique",
    )
    _positive_int(runtime, "answer_batch_size", 8192, "runtime.answer_batch_size")
    _positive_int(runtime, "scoring_chunk_size", 4096, "runtime.scoring_chunk_size")
    if _finite_float(debug, "residual_drift_tolerance", 1.0e-5, "debug.residual_drift_tolerance") < 0.0:
        raise ValueError("debug.residual_drift_tolerance must be non-negative")
    if _finite_float(debug, "loss_tolerance", 1.0e-4, "debug.loss_tolerance") < 0.0:
        raise ValueError("debug.loss_tolerance must be non-negative")

    measurement_mode = str(privacy.get("measurement_mode", "static_all")).lower()
    if measurement_mode != "static_all":
        raise NotImplementedError(
            "Only privacy.measurement_mode=static_all is implemented. "
            "adaptive_select_measure/select-measure-generate is not implemented in this version."
        )
    reuse_from = measurement.get("reuse_from", measurement.get("artifact_dir"))
    if reuse_from is not None and str(reuse_from) == "":
        raise ValueError("measurement.reuse_from must be a non-empty path when provided")
    transcript_only_generation = bool(run.get("transcript_only_generation", False))
    if transcript_only_generation:
        if not dp_release_mode:
            raise ValueError(
                "run.transcript_only_generation=true requires privacy.dp_release_mode=true"
            )
        if run.get("input_csv") not in {None, ""}:
            raise ValueError(
                "run.transcript_only_generation=true forbids run.input_csv"
            )
        if reuse_from is None:
            raise ValueError(
                "run.transcript_only_generation=true requires measurement.reuse_from"
            )
        if not bool(workload.get("reuse_from_measurement", False)):
            raise ValueError(
                "run.transcript_only_generation=true requires "
                "workload.reuse_from_measurement=true"
            )
        if init.get("encoded_npy") not in {None, ""}:
            raise ValueError(
                "run.transcript_only_generation=true forbids init.encoded_npy; "
                "initialization must use only the released transcript"
            )
    fission = measurement.get("fission", {})
    if fission is None:
        fission = {}
    if not isinstance(fission, dict):
        raise ValueError("measurement.fission must be a mapping")
    if bool(fission.get("enabled", False)):
        if privacy_mode != "dp":
            raise ValueError("measurement.fission.enabled currently requires privacy.mode='dp'")
        train_fraction = _finite_float(
            fission,
            "train_fraction",
            0.8,
            "measurement.fission.train_fraction",
        )
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("measurement.fission.train_fraction must be in (0, 1)")
        _positive_int(
            fission,
            "checkpoint_interval",
            100,
            "measurement.fission.checkpoint_interval",
        )
        _positive_int(
            fission,
            "seed_offset",
            51_771,
            "measurement.fission.seed_offset",
            allow_zero=True,
        )
        one_se_multiplier = _finite_float(
            fission,
            "one_se_multiplier",
            1.0,
            "measurement.fission.one_se_multiplier",
        )
        if one_se_multiplier < 0.0:
            raise ValueError("measurement.fission.one_se_multiplier must be non-negative")
        _validate_choice(
            fission,
            "selection_rule",
            {
                "earliest_within_one_se",
                "validation_minimum",
                "l2_one_se_tvd_minimum",
                "l2_one_se_tvd_upper_minimum",
            },
            "earliest_within_one_se",
            "measurement.fission.selection_rule",
        )
        _validate_choice(
            fission,
            "optimization_branch",
            {"train", "validation"},
            "train",
            "measurement.fission.optimization_branch",
        )
        uncertainty = projection.get("uncertainty", {})
        if uncertainty is not None and not isinstance(uncertainty, dict):
            raise ValueError("projection.uncertainty must be a mapping")
        if isinstance(uncertainty, dict) and bool(uncertainty.get("enabled", False)):
            raise ValueError(
                "measurement.fission currently requires projection.uncertainty.enabled=false"
            )

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

    if dp_release_mode:
        forbidden_evaluation = [
            name
            for name, default in (
                ("compute_true_query_error", True),
                ("compute_heldout_query_error", False),
                ("downstream_ml", False),
            )
            if bool(evaluation.get(name, default))
        ]
        if isinstance(oracle_bias, dict) and bool(oracle_bias.get("enabled", False)):
            forbidden_evaluation.append("oracle_projection_bias.enabled")
        if forbidden_evaluation:
            raise ValueError(
                "privacy.dp_release_mode=true forbids in-process private evaluation: "
                + ", ".join(forbidden_evaluation)
            )

    if bool(evaluation.get("downstream_ml", False)):
        raise NotImplementedError("evaluation.downstream_ml=true is not implemented")

    candidate_backend = str(qdte.get("candidate_backend", "cpu_repair"))
    score_backend = str(qdte.get("score_backend", "dense_gpu"))
    candidate_compiler = str(qdte.get("candidate_compiler", "single_query"))
    if "eval_every" in qdte:
        raise ValueError(
            "qdte.eval_every is not implemented; use qdte.log_every for public objective logging "
            "and keep true-answer evaluation offline"
        )
    if score_backend == "sparse_delta_gpu" and candidate_backend not in {"jax_repair", "gpu_repair"}:
        raise ValueError("qdte.score_backend='sparse_delta_gpu' requires qdte.candidate_backend to be jax_repair or gpu_repair")
    if candidate_backend in {"jax_repair", "gpu_repair"} and score_backend not in {
        "dense_gpu",
        "sparse_delta_gpu",
    }:
        raise ValueError(
            "GPU candidate backends use fused scoring and require "
            "qdte.score_backend to be dense_gpu or sparse_delta_gpu"
        )
    if candidate_compiler != "single_query" and candidate_backend in {"jax_repair", "gpu_repair"}:
        raise NotImplementedError(
            "qdte.candidate_compiler other than 'single_query' currently requires qdte.candidate_backend=cpu_repair"
        )

    _positive_int(qdte, "max_iters", 5000, "qdte.max_iters", allow_zero=True)
    _positive_int(qdte, "num_active_targets", 64, "qdte.num_active_targets")
    _positive_int(qdte, "candidates_per_target", 64, "qdte.candidates_per_target")
    _positive_int(qdte, "total_candidates_per_iter", 4096, "qdte.total_candidates_per_iter")
    _positive_int(qdte, "source_over_sample_factor", 8, "qdte.source_over_sample_factor")
    _positive_int(qdte, "full_recompute_every", 0, "qdte.full_recompute_every", allow_zero=True)
    _positive_int(qdte, "stop_patience", 50, "qdte.stop_patience")
    _positive_int(qdte, "log_every", 100, "qdte.log_every")
    _positive_int(qdte, "gpu_return_top_k", 0, "qdte.gpu_return_top_k", allow_zero=True)
    _positive_int(qdte, "gpu_return_oversample_factor", 2, "qdte.gpu_return_oversample_factor")
    _positive_int(qdte, "gpu_source_draws", 8, "qdte.gpu_source_draws")
    _positive_int(qdte, "gpu_batches_per_iter", 1, "qdte.gpu_batches_per_iter")
    _positive_int(
        qdte,
        "gpu_score_query_block_size",
        0,
        "qdte.gpu_score_query_block_size",
        allow_zero=True,
    )
    _positive_int(qdte, "gpu_sparse_query_block_size", 64, "qdte.gpu_sparse_query_block_size")
    _positive_int(qdte, "gpu_sparse_changed_attr_capacity", 1, "qdte.gpu_sparse_changed_attr_capacity")
    if _finite_float(qdte, "kappa_noise", 1.0, "qdte.kappa_noise") < 0.0:
        raise ValueError("qdte.kappa_noise must be non-negative")
    if _finite_float(qdte, "lambda_cost", 0.01, "qdte.lambda_cost") < 0.0:
        raise ValueError("qdte.lambda_cost must be non-negative")
    if _finite_float(qdte, "numerical_distance_gamma", 0.1, "qdte.numerical_distance_gamma") < 0.0:
        raise ValueError("qdte.numerical_distance_gamma must be non-negative")
    random_fraction = _finite_float(qdte, "random_candidate_fraction", 0.05, "qdte.random_candidate_fraction")
    if random_fraction < 0.0 or random_fraction > 1.0:
        raise ValueError("qdte.random_candidate_fraction must be in [0, 1]")
    if _finite_float(qdte, "min_advantage", 1.0e-6, "qdte.min_advantage") < 0.0:
        raise ValueError("qdte.min_advantage must be non-negative")
    if _finite_float(qdte, "debt_alpha", 0.0, "qdte.debt_alpha") < 0.0:
        raise ValueError("qdte.debt_alpha must be non-negative")
    debt_decay = _finite_float(qdte, "debt_decay", 0.95, "qdte.debt_decay")
    if debt_decay < 0.0 or debt_decay > 1.0:
        raise ValueError("qdte.debt_decay must be in [0, 1]")
    if _finite_float(qdte, "debt_repay", 1.0, "qdte.debt_repay") < 0.0:
        raise ValueError("qdte.debt_repay must be non-negative")
    if _finite_float(qdte, "debt_cap", 1.0e6, "qdte.debt_cap") <= 0.0:
        raise ValueError("qdte.debt_cap must be positive")

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
            if method == "query_space_feasible_lsq":
                if float(consistency.get("certificate_feasibility_tolerance", 1.0e-6)) < 0.0:
                    raise ValueError(
                        "projection.consistency.certificate_feasibility_tolerance must be non-negative"
                    )
                if float(consistency.get("certificate_gap_absolute_tolerance", 1.0e-7)) < 0.0:
                    raise ValueError(
                        "projection.consistency.certificate_gap_absolute_tolerance must be non-negative"
                    )
                if float(consistency.get("certificate_gap_relative_tolerance", 1.0e-8)) < 0.0:
                    raise ValueError(
                        "projection.consistency.certificate_gap_relative_tolerance must be non-negative"
                    )
                if int(consistency.get("certificate_max_iterations", 1_000)) <= 0:
                    raise ValueError(
                        "projection.consistency.certificate_max_iterations must be positive"
                    )
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
    n_syn = init.get("N_syn", "same_as_real")
    if n_syn is not None and str(n_syn) != "same_as_real" and int(n_syn) <= 0:
        raise ValueError("init.N_syn must be positive or 'same_as_real'")

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
        {"dense_gpu", "target_only", "sparse_delta", "sparse_delta_gpu", "precision_operator"},
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
        "objective_weight_profile",
        {"none", "joint_utility_envelope"},
        "none",
        "qdte.objective_weight_profile",
    )
    _validate_choice(
        qdte,
        "objective_loss",
        {"quadratic", "tvd_l1"},
        "quadratic",
        "qdte.objective_loss",
    )
    _validate_choice(
        qdte,
        "precision_operator",
        {
            "diagonal",
            "coarsened_interaction_full_covariance",
            "orthogonal_interaction",
            "orthogonal_interaction_shrink_raw",
            "orthogonal_interaction_shrink_analytic",
            "orthogonal_interaction_p3_raw",
            "orthogonal_interaction_p3_bootdiag",
            "orthogonal_interaction_p3_active_set",
        },
        "diagonal",
        "qdte.precision_operator",
    )
    objective_loss = str(qdte.get("objective_loss", "quadratic"))
    objective_weighting = str(qdte.get("objective_weighting", "variance"))
    objective_weight_profile = str(qdte.get("objective_weight_profile", "none"))
    precision_operator = str(qdte.get("precision_operator", "diagonal"))
    structured_swap_enabled = bool(qdte.get("structured_swap_enabled", False))
    _validate_choice(
        qdte,
        "structured_swap_compiler",
        {"global_random_v1", "interaction_rectangle_v1"},
        "global_random_v1",
        "qdte.structured_swap_compiler",
    )
    structured_swap_compiler = str(
        qdte.get("structured_swap_compiler", "global_random_v1")
    )
    confidence_stop = qdte.get("confidence_stop", {})
    if confidence_stop is None:
        confidence_stop = {}
    if not isinstance(confidence_stop, dict):
        raise ValueError("qdte.confidence_stop must be a mapping")
    if "enabled" in confidence_stop and not isinstance(confidence_stop["enabled"], bool):
        raise ValueError("qdte.confidence_stop.enabled must be boolean")
    confidence_stop_enabled = bool(confidence_stop.get("enabled", False))
    _validate_choice(
        confidence_stop,
        "method",
        {"chi_square"},
        "chi_square",
        "qdte.confidence_stop.method",
    )
    confidence_stop_alpha = _finite_float(
        confidence_stop,
        "alpha",
        0.05,
        "qdte.confidence_stop.alpha",
    )
    if not 0.0 < confidence_stop_alpha < 1.0:
        raise ValueError("qdte.confidence_stop.alpha must be in (0, 1)")
    entropy = qdte.get("entropy", {})
    if entropy is None:
        entropy = {}
    if not isinstance(entropy, dict):
        raise ValueError("qdte.entropy must be a mapping")
    if "enabled" in entropy and not isinstance(entropy["enabled"], bool):
        raise ValueError("qdte.entropy.enabled must be boolean")
    entropy_enabled = bool(entropy.get("enabled", False))
    _validate_choice(
        entropy,
        "method",
        {"confidence_constrained_product_kl_primal_dual_v1"},
        "confidence_constrained_product_kl_primal_dual_v1",
        "qdte.entropy.method",
    )
    entropy_alpha = _finite_float(entropy, "alpha", 0.05, "qdte.entropy.alpha")
    entropy_smoothing = _finite_float(
        entropy,
        "product_prior_smoothing",
        1.0,
        "qdte.entropy.product_prior_smoothing",
    )
    entropy_dual_initial = _finite_float(
        entropy,
        "dual_initial",
        1.0,
        "qdte.entropy.dual_initial",
    )
    if entropy_enabled:
        if entropy_alpha != 0.05:
            raise ValueError("The frozen first entropy profile requires qdte.entropy.alpha=0.05")
        if entropy_smoothing != 1.0:
            raise ValueError(
                "The frozen first entropy profile requires qdte.entropy.product_prior_smoothing=1.0"
            )
        if entropy_dual_initial != 1.0:
            raise ValueError(
                "The frozen first entropy profile requires qdte.entropy.dual_initial=1.0"
            )
    rce = qdte.get("rce", {})
    if rce is None:
        rce = {}
    if not isinstance(rce, dict):
        raise ValueError("qdte.rce must be a mapping")
    if "enabled" in rce and not isinstance(rce["enabled"], bool):
        raise ValueError("qdte.rce.enabled must be boolean")
    rce_enabled = bool(rce.get("enabled", False))
    _validate_choice(
        rce,
        "method",
        {"row_realizable_confidence_set_entropic_primal_dual_v1"},
        "row_realizable_confidence_set_entropic_primal_dual_v1",
        "qdte.rce.method",
    )
    _validate_choice(
        rce,
        "reference_prior",
        {"released_product", "released_confidence_forest_v1"},
        "released_product",
        "qdte.rce.reference_prior",
    )
    _validate_choice(
        rce,
        "prefix_backend",
        {"feature_cpu_exact", "feature_gpu"},
        "feature_cpu_exact",
        "qdte.rce.prefix_backend",
    )
    rce_reference_prior = str(rce.get("reference_prior", "released_product"))
    streamwise_cdwf = rce.get("streamwise_cdwf", {})
    if streamwise_cdwf is None:
        streamwise_cdwf = {}
    if not isinstance(streamwise_cdwf, dict):
        raise ValueError("qdte.rce.streamwise_cdwf must be a mapping")
    if "enabled" in streamwise_cdwf and not isinstance(
        streamwise_cdwf["enabled"], bool
    ):
        raise ValueError("qdte.rce.streamwise_cdwf.enabled must be boolean")
    streamwise_cdwf_enabled = bool(streamwise_cdwf.get("enabled", False))
    c0_diagnostic = rce.get("c0_diagnostic", {})
    if c0_diagnostic is None:
        c0_diagnostic = {}
    if not isinstance(c0_diagnostic, dict):
        raise ValueError("qdte.rce.c0_diagnostic must be a mapping")
    if "enabled" in c0_diagnostic and not isinstance(
        c0_diagnostic["enabled"], bool
    ):
        raise ValueError("qdte.rce.c0_diagnostic.enabled must be boolean")
    c0_enabled = bool(c0_diagnostic.get("enabled", False))
    _validate_choice(
        c0_diagnostic,
        "support_mode",
        {"released", "oracle"},
        "released",
        "qdte.rce.c0_diagnostic.support_mode",
    )
    _validate_choice(
        c0_diagnostic,
        "interaction_center",
        {"dp", "clean"},
        "dp",
        "qdte.rce.c0_diagnostic.interaction_center",
    )
    precision_homotopy = rce.get("precision_homotopy_diagnostic", {})
    if precision_homotopy is None:
        precision_homotopy = {}
    if not isinstance(precision_homotopy, dict):
        raise ValueError("qdte.rce.precision_homotopy_diagnostic must be a mapping")
    if "enabled" in precision_homotopy and not isinstance(
        precision_homotopy["enabled"], bool
    ):
        raise ValueError(
            "qdte.rce.precision_homotopy_diagnostic.enabled must be boolean"
        )
    precision_homotopy_enabled = bool(precision_homotopy.get("enabled", False))
    homotopy_gamma = precision_homotopy.get("gamma", 1.0)
    if isinstance(homotopy_gamma, str):
        if homotopy_gamma.strip().lower() != "infinity":
            raise ValueError(
                "precision homotopy gamma must be one of 1, 2, 4, 8, infinity"
            )
    elif float(homotopy_gamma) not in {1.0, 2.0, 4.0, 8.0}:
        raise ValueError(
            "precision homotopy gamma must be one of 1, 2, 4, 8, infinity"
        )
    if c0_enabled and precision_homotopy_enabled:
        raise ValueError("C0 and precision homotopy diagnostics are mutually exclusive")
    if streamwise_cdwf_enabled and (c0_enabled or precision_homotopy_enabled):
        raise ValueError(
            "C3 streamwise CDWF cannot compose C0 or precision-homotopy diagnostics"
        )
    rce_alpha_l2 = _finite_float(rce, "alpha_l2", 0.025, "qdte.rce.alpha_l2")
    rce_alpha_linf = _finite_float(rce, "alpha_linf", 0.025, "qdte.rce.alpha_linf")
    rce_smoothing = _finite_float(
        rce,
        "product_prior_smoothing",
        1.0,
        "qdte.rce.product_prior_smoothing",
    )
    if rce_enabled:
        if rce_alpha_l2 != 0.025 or rce_alpha_linf != 0.025:
            raise ValueError(
                "The frozen RCE-v1 profile requires alpha_l2=alpha_linf=0.025"
            )
        if rce_smoothing != 1.0:
            raise ValueError(
                "The frozen RCE-v1 profile requires qdte.rce.product_prior_smoothing=1.0"
            )
    if bool(fission.get("enabled", False)):
        supported_fission_objective = (
            objective_loss == "quadratic" and objective_weighting == "variance"
        ) or (
            objective_loss == "tvd_l1" and objective_weighting == "unweighted"
        )
        if not supported_fission_objective:
            raise ValueError(
                "measurement.fission requires either variance-weighted quadratic "
                "or unweighted tvd_l1 optimization"
            )
    transport_mode = str(qdte.get("transport_mode", "microbatch_greedy"))
    atom_flow_update_mode = str(qdte.get("atom_flow_update_mode", "batch"))
    if score_backend == "precision_operator":
        if candidate_backend != "cpu_repair":
            raise ValueError("qdte.score_backend='precision_operator' requires qdte.candidate_backend='cpu_repair'")
        if objective_loss != "quadratic":
            raise ValueError("qdte.score_backend='precision_operator' requires qdte.objective_loss='quadratic'")
        if transport_mode != "atom_flow" or atom_flow_update_mode != "batch":
            raise ValueError(
                "qdte.score_backend='precision_operator' requires batch atom_flow transport"
            )
        if structured_swap_enabled and not (
            precision_operator == "orthogonal_interaction"
            and structured_swap_compiler == "interaction_rectangle_v1"
        ):
            raise ValueError(
                "precision-operator structured_swap requires the exact raw "
                "orthogonal_interaction precision and interaction_rectangle_v1 compiler"
            )
        if bool(qdte.get("candidate_diagnostics", False)):
            raise ValueError("precision-operator scoring does not yet support candidate_diagnostics")
        if bool(fission.get("enabled", False)):
            raise ValueError("precision-operator scoring does not yet support measurement fission")
    elif precision_operator != "diagonal":
        raise ValueError(
            "non-diagonal qdte.precision_operator requires "
            "qdte.score_backend='precision_operator'"
        )
    if precision_operator in {
        "coarsened_interaction_full_covariance",
        "orthogonal_interaction",
        "orthogonal_interaction_shrink_raw",
        "orthogonal_interaction_shrink_analytic",
        "orthogonal_interaction_p3_raw",
        "orthogonal_interaction_p3_bootdiag",
        "orthogonal_interaction_p3_active_set",
    }:
        if objective_weighting != "variance" or objective_weight_profile != "none":
            raise ValueError(
                "orthogonal interaction precision requires variance weighting and no query weight profile"
            )
    if confidence_stop_enabled and precision_operator != "orthogonal_interaction":
        raise ValueError(
            "qdte.confidence_stop.enabled=true currently requires "
            "qdte.precision_operator='orthogonal_interaction'"
        )
    if entropy_enabled:
        if precision_operator != "orthogonal_interaction":
            raise ValueError(
                "qdte.entropy.enabled=true requires qdte.precision_operator='orthogonal_interaction'"
            )
        if score_backend != "precision_operator" or candidate_backend != "cpu_repair":
            raise ValueError(
                "qdte.entropy.enabled=true requires precision_operator scoring and cpu_repair candidates"
            )
        if objective_loss != "quadratic" or objective_weighting != "variance":
            raise ValueError(
                "qdte.entropy.enabled=true requires the variance-weighted quadratic objective"
            )
        if transport_mode != "atom_flow" or atom_flow_update_mode != "batch":
            raise ValueError("qdte.entropy.enabled=true requires batch atom_flow transport")
        if structured_swap_enabled:
            raise ValueError("The first entropy profile does not support structured swaps")
        if confidence_stop_enabled:
            raise ValueError("Disable qdte.confidence_stop when qdte.entropy is enabled")
        if bool(fission.get("enabled", False)):
            raise ValueError("The first entropy profile does not support measurement fission")
        if not bool(qdte.get("allow_below_noise_fallback", False)):
            raise ValueError(
                "qdte.entropy.enabled=true requires qdte.allow_below_noise_fallback=true"
            )
    if rce_enabled:
        if precision_operator not in {
            "orthogonal_interaction",
            "coarsened_interaction_full_covariance",
        }:
            raise ValueError(
                "qdte.rce.enabled=true requires precision_operator="
                "'orthogonal_interaction' or "
                "'coarsened_interaction_full_covariance'"
            )
        if (
            precision_operator == "coarsened_interaction_full_covariance"
            and rce_reference_prior != "released_confidence_forest_v1"
        ):
            raise ValueError(
                "coarsened interaction RCE requires the released confidence forest prior"
            )
        if (
            precision_operator == "coarsened_interaction_full_covariance"
            and str(rce.get("prefix_backend", "feature_cpu_exact"))
            != "feature_cpu_exact"
        ):
            raise ValueError(
                "coarsened interaction RCE requires exact CPU full-covariance prefix scoring"
            )
        if score_backend != "precision_operator" or candidate_backend != "cpu_repair":
            raise ValueError(
                "qdte.rce.enabled=true requires precision_operator scoring and cpu_repair candidates"
            )
        if objective_loss != "quadratic" or objective_weighting != "variance":
            raise ValueError(
                "qdte.rce.enabled=true requires the variance-weighted quadratic base objective"
            )
        if transport_mode != "atom_flow" or atom_flow_update_mode != "batch":
            raise ValueError("qdte.rce.enabled=true requires batch atom_flow transport")
        if structured_swap_enabled:
            raise ValueError("The frozen RCE-v1 profile does not support structured swaps")
        if confidence_stop_enabled:
            raise ValueError("RCE uses its own mixed confidence set; disable confidence_stop")
        if entropy_enabled:
            raise ValueError("RCE-v1 and the legacy entropy controller are mutually exclusive")
        if bool(fission.get("enabled", False)):
            raise ValueError("The frozen RCE-v1 profile does not compose Gaussian fission")
    if streamwise_cdwf_enabled:
        if not rce_enabled:
            raise ValueError("C3 streamwise CDWF requires qdte.rce.enabled=true")
        if precision_operator != "orthogonal_interaction":
            raise ValueError(
                "C3 streamwise CDWF requires the exact orthogonal_interaction precision"
            )
        if rce_reference_prior != "released_confidence_forest_v1":
            raise ValueError(
                "C3 streamwise CDWF requires the released confidence forest prior"
            )
        if not transcript_only_generation:
            raise ValueError(
                "C3 final generation must run transcript-only from a sealed public artifact"
            )
        if reuse_from is None or not bool(workload.get("reuse_from_measurement", False)):
            raise ValueError(
                "C3 streamwise CDWF requires a reused measurement artifact and workload"
            )
    if c0_enabled:
        if not rce_enabled or rce_reference_prior != "released_confidence_forest_v1":
            raise ValueError(
                "C0 diagnostic requires enabled RCE with the released confidence forest prior"
            )
        if privacy_mode != "oracle" or dp_release_mode:
            raise ValueError(
                "C0 diagnostic is offline-only and requires privacy.mode='oracle' with "
                "privacy.dp_release_mode=false"
            )
        if transcript_only_generation:
            raise ValueError(
                "C0 diagnostic must load the private table explicitly; transcript-only "
                "generation is reserved for DP post-processing"
            )
        if run.get("input_csv") in {None, ""}:
            raise ValueError("C0 diagnostic requires run.input_csv")
        if reuse_from is None or not bool(workload.get("reuse_from_measurement", False)):
            raise ValueError(
                "C0 diagnostic requires a frozen measurement artifact and workload"
            )
    if precision_homotopy_enabled:
        if not rce_enabled or rce_reference_prior != "released_confidence_forest_v1":
            raise ValueError(
                "Precision homotopy requires enabled RCE with the CCF prior"
            )
        if privacy_mode != "oracle" or dp_release_mode:
            raise ValueError(
                "Precision homotopy is offline-only and requires oracle mode"
            )
        if transcript_only_generation:
            raise ValueError(
                "Precision homotopy must explicitly load its private diagnostic table"
            )
        if run.get("input_csv") in {None, ""}:
            raise ValueError("Precision homotopy requires run.input_csv")
        if reuse_from is None or not bool(workload.get("reuse_from_measurement", False)):
            raise ValueError(
                "Precision homotopy requires a frozen measurement artifact and workload"
            )
    if objective_loss == "tvd_l1":
        if candidate_backend != "cpu_repair":
            raise ValueError("qdte.objective_loss='tvd_l1' currently requires qdte.candidate_backend='cpu_repair'")
        if transport_mode != "atom_flow" or atom_flow_update_mode != "batch":
            raise ValueError(
                "qdte.objective_loss='tvd_l1' currently requires "
                "qdte.transport_mode='atom_flow' and qdte.atom_flow_update_mode='batch'"
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
        "structured_swap_accept_schedule",
        {"fixed", "none", "linear", "cosine", "exponential"},
        "cosine",
        "qdte.structured_swap_accept_schedule",
    )
    _validate_choice(
        qdte,
        "structured_swap_delta_backend",
        {"sparse_cpu", "dense_gpu"},
        "sparse_cpu",
        "qdte.structured_swap_delta_backend",
    )
    structured_swap_delta_backend = str(
        qdte.get("structured_swap_delta_backend", "sparse_cpu")
    )
    if structured_swap_enabled and structured_swap_compiler == "interaction_rectangle_v1":
        if (
            precision_operator != "orthogonal_interaction"
            or score_backend != "precision_operator"
            or candidate_backend != "cpu_repair"
            or objective_loss != "quadratic"
            or objective_weighting != "variance"
            or objective_weight_profile != "none"
            or transport_mode != "atom_flow"
            or atom_flow_update_mode != "batch"
        ):
            raise ValueError(
                "interaction_rectangle_v1 requires the unshrunk raw orthogonal-interaction "
                "quadratic objective with precision-operator scoring and batch atom-flow transport"
            )
        if structured_swap_delta_backend != "sparse_cpu":
            raise ValueError(
                "interaction_rectangle_v1 requires qdte.structured_swap_delta_backend='sparse_cpu'"
            )
        if confidence_stop_enabled:
            raise ValueError(
                "The first interaction-cycle profile does not support confidence stopping"
            )
        if entropy_enabled:
            raise ValueError("The first interaction-cycle profile does not support entropy")
        if bool(fission.get("enabled", False)):
            raise ValueError(
                "The first interaction-cycle profile does not support measurement fission"
            )
    _validate_choice(
        qdte,
        "structured_swap_noise_guard_mode",
        {"fixed", "bonferroni"},
        "fixed",
        "qdte.structured_swap_noise_guard_mode",
    )
    _validate_choice(
        qdte,
        "transport_prefix_strategy",
        {"largest_positive", "best_advantage"},
        "largest_positive",
        "qdte.transport_prefix_strategy",
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
    for key in (
        "structured_swap_start_iter",
        "structured_swap_interval",
        "structured_swap_candidate_units",
        "structured_swap_transport_pool",
        "structured_swap_accept_start",
        "structured_swap_accept_end",
    ):
        if key in qdte and int(qdte[key]) <= 0:
            raise ValueError(f"qdte.{key} must be positive")
    if (
        "structured_swap_accept_start" in qdte
        and "structured_swap_accept_end" in qdte
        and int(qdte["structured_swap_accept_start"]) < int(qdte["structured_swap_accept_end"])
    ):
        raise ValueError("qdte.structured_swap_accept_start must be at least structured_swap_accept_end")
    for key in (
        "structured_swap_noise_guard_kappa",
        "structured_swap_trigger_rms",
        "structured_swap_policy_prior_strength",
    ):
        if key in qdte and float(qdte[key]) < 0.0:
            raise ValueError(f"qdte.{key} must be non-negative")
    if "structured_swap_noise_guard_alpha" in qdte and not 0.0 < float(
        qdte["structured_swap_noise_guard_alpha"]
    ) < 1.0:
        raise ValueError("qdte.structured_swap_noise_guard_alpha must be in (0, 1)")
    if "structured_swap_exploration_floor" in qdte and not 0.0 <= float(
        qdte["structured_swap_exploration_floor"]
    ) <= 1.0:
        raise ValueError("qdte.structured_swap_exploration_floor must be in [0, 1]")
    if "structured_swap_policy_decay" in qdte and not 0.0 <= float(
        qdte["structured_swap_policy_decay"]
    ) < 1.0:
        raise ValueError("qdte.structured_swap_policy_decay must be in [0, 1)")
    if bool(qdte.get("structured_swap_enabled", False)) and objective_loss == "tvd_l1":
        if float(qdte.get("structured_swap_noise_guard_kappa", 2.0)) != 0.0:
            raise ValueError(
                "structured TVD-L1 currently requires qdte.structured_swap_noise_guard_kappa=0"
            )
        if str(qdte.get("structured_swap_noise_guard_mode", "fixed")) != "fixed":
            raise ValueError(
                "structured TVD-L1 currently requires qdte.structured_swap_noise_guard_mode='fixed'"
            )
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
    if (
        int(qdte.get("random_group_max_size", 0)) > 0
        and int(qdte.get("random_group_max_size", 0)) < int(qdte.get("random_group_min_size", 2))
    ):
        raise ValueError("qdte.random_group_max_size must be zero or at least random_group_min_size")
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
    if (
        int(qdte.get("directed_group_max_size", 0)) > 0
        and int(qdte.get("directed_group_max_size", 0)) < int(qdte.get("directed_group_min_size", 1))
    ):
        raise ValueError("qdte.directed_group_max_size must be zero or at least directed_group_min_size")
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
    if (
        int(qdte.get("directed_group_augment_max_size", 0)) > 0
        and int(qdte.get("directed_group_augment_max_size", 0))
        < int(qdte.get("directed_group_augment_min_size", qdte.get("directed_group_min_size", 1)))
    ):
        raise ValueError(
            "qdte.directed_group_augment_max_size must be zero or at least directed_group_augment_min_size"
        )
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
