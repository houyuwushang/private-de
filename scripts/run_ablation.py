#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdte.config import apply_overrides, load_yaml, set_nested
from qdte.evolution.engine import run_qdte


def _drop_output_dir_override(overrides: list[str]) -> list[str]:
    filtered: list[str] = []
    i = 0
    while i < len(overrides):
        token = overrides[i]
        if not token.startswith("--"):
            filtered.append(token)
            i += 1
            continue
        key_value = token[2:]
        if "=" in key_value:
            key, _ = key_value.split("=", 1)
            if key != "run.output_dir":
                filtered.append(token)
            i += 1
        else:
            key = key_value
            if key != "run.output_dir":
                filtered.extend(overrides[i : i + 2])
            i += 2
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    requested_variant = args.variant
    variant = requested_variant
    blind_accept = False
    if variant.startswith("blind_"):
        blind_accept = True
        variant = variant[len("blind_") :]
    out = Path(config.get("run", {}).get("output_dir", "outputs/qdte_run"))
    set_nested(config, "run.output_dir", str(out.parent / f"{out.name}_{requested_variant}"))
    if variant in {"single_query", "baseline"}:
        pass
    elif variant == "masked_single_query":
        set_nested(config, "qdte.candidate_compiler", "masked_single_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "relaxed_masked_single_query":
        set_nested(config, "qdte.candidate_compiler", "relaxed_masked_single_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "random_mutation":
        set_nested(config, "qdte.random_candidate_fraction", 1.0)
    elif variant in {"random_best_edit", "random_best_mutation", "private_gsd_mutate", "pgsd_mutate"}:
        set_nested(config, "qdte.random_candidate_fraction", 1.0)
        set_nested(config, "qdte.accepted_per_iter", 1)
        set_nested(config, "qdte.transport_mode", "microbatch_greedy")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
    elif variant == "random_group_advantage":
        set_nested(config, "qdte.random_candidate_fraction", 1.0)
        set_nested(config, "qdte.transport_mode", "random_group")
        set_nested(config, "qdte.random_group_count", 128)
        set_nested(config, "qdte.random_group_min_size", 2)
        set_nested(config, "qdte.random_group_max_size", 0)
        set_nested(config, "qdte.random_group_pool_multiplier", 0)
        set_nested(config, "qdte.random_group_max_pool", 0)
    elif variant == "directed_group_advantage":
        set_nested(config, "qdte.random_candidate_fraction", 1.0)
        set_nested(config, "qdte.transport_mode", "directed_group")
        set_nested(config, "qdte.directed_group_seed_count", 64)
        set_nested(config, "qdte.directed_group_min_size", 1)
        set_nested(config, "qdte.directed_group_max_size", 8)
        set_nested(config, "qdte.directed_group_pool_multiplier", 0)
        set_nested(config, "qdte.directed_group_max_pool", 0)
        set_nested(config, "qdte.directed_group_allow_negative_steps", False)
    elif variant in {"pgsd_style_mutate50", "private_gsd_mutate50"}:
        set_nested(config, "qdte.random_candidate_fraction", 1.0)
        set_nested(config, "qdte.total_candidates_per_iter", 50)
        set_nested(config, "qdte.candidates_per_target", 50)
        set_nested(config, "qdte.num_active_targets", 1)
        set_nested(config, "qdte.accepted_per_iter", 1)
        set_nested(config, "qdte.transport_mode", "microbatch_greedy")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.kappa_noise", 0.0)
    elif variant == "no_edit_cost":
        set_nested(config, "qdte.lambda_cost", 0.0)
    elif variant == "sequential_greedy":
        set_nested(config, "qdte.accepted_per_iter", 1)
        set_nested(config, "qdte.transport_mode", "sequential_greedy")
    elif variant == "target_only":
        set_nested(config, "qdte.score_backend", "target_only")
    elif variant == "no_threshold":
        set_nested(config, "qdte.kappa_noise", 0.0)
    elif variant == "paired_query":
        set_nested(config, "qdte.candidate_compiler", "paired_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.paired_candidate_fraction", 0.25)
        set_nested(config, "qdte.paired_try_break_source", True)
    elif variant == "paired_query_full":
        set_nested(config, "qdte.candidate_compiler", "paired_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.paired_candidate_fraction", 1.0)
        set_nested(config, "qdte.paired_try_break_source", True)
    elif variant == "masked_paired_query":
        set_nested(config, "qdte.candidate_compiler", "masked_paired_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.paired_candidate_fraction", 0.25)
        set_nested(config, "qdte.paired_try_break_source", True)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "masked_paired_query_full":
        set_nested(config, "qdte.candidate_compiler", "masked_paired_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.paired_candidate_fraction", 1.0)
        set_nested(config, "qdte.paired_try_break_source", True)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "masked_exit_query":
        set_nested(config, "qdte.candidate_compiler", "masked_exit_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.paired_candidate_fraction", 0.25)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "masked_exit_query_full":
        set_nested(config, "qdte.candidate_compiler", "masked_exit_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.paired_candidate_fraction", 1.0)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "directed_exit_only":
        set_nested(config, "qdte.candidate_compiler", "directed_exit_only")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
    elif variant == "masked_exit_only":
        set_nested(config, "qdte.candidate_compiler", "masked_exit_only")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.mask_min_terms", 1)
        set_nested(config, "qdte.mask_max_terms", 2)
    elif variant == "random_source_directed_exit":
        set_nested(config, "qdte.candidate_compiler", "random_source_directed_exit")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
    elif variant == "residual_weighted_mutation":
        set_nested(config, "qdte.candidate_compiler", "residual_weighted_mutation")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.residual_value_uniform_mix", 0.15)
        set_nested(config, "qdte.residual_attr_uniform_mix", 0.15)
    elif variant == "enumerated_local":
        set_nested(config, "qdte.candidate_compiler", "enumerated_local")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.enumerated_values_per_attr", 0)
    elif variant == "soft_single_query":
        set_nested(config, "qdte.candidate_compiler", "soft_single_query")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.soft_source_match_weight", 4.0)
        set_nested(config, "qdte.soft_source_mismatch_weight", 1.0)
    elif variant == "residual_value_mutation":
        set_nested(config, "qdte.candidate_compiler", "residual_value_mutation")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.residual_value_uniform_mix", 0.15)
    elif variant == "proposal_mixture":
        set_nested(config, "qdte.candidate_compiler", "proposal_mixture")
        set_nested(config, "qdte.random_candidate_fraction", 0.0)
        set_nested(config, "qdte.mixture_random_fraction", 0.20)
        set_nested(config, "qdte.mixture_residual_weighted_fraction", 0.25)
        set_nested(config, "qdte.mixture_enumerated_fraction", 0.20)
        set_nested(config, "qdte.mixture_soft_single_fraction", 0.20)
        set_nested(config, "qdte.mixture_residual_value_fraction", 0.15)
    elif variant == "qdte_mixture":
        set_nested(config, "qdte.candidate_compiler", "qdte_mixture")
        set_nested(config, "qdte.random_candidate_fraction", 0.125)
        set_nested(config, "qdte.candidate_shortfall_policy", "random")
        set_nested(config, "qdte.qdte_mixture_single_fraction", 0.45)
        set_nested(config, "qdte.qdte_mixture_masked_single_fraction", 0.20)
        set_nested(config, "qdte.qdte_mixture_enumerated_fraction", 0.25)
        set_nested(config, "qdte.qdte_mixture_relaxed_masked_fraction", 0.10)
    elif variant in {"constructive_pair", "constructive_pair_accept_anneal", "constructive_pair_accept_anneal32"}:
        set_nested(config, "qdte.candidate_compiler", "single_query")
        set_nested(config, "qdte.transport_mode", "constructive_pair")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.constructive_pair_partner_limit", 16)
        set_nested(config, "qdte.constructive_pair_harm_query_limit", 16)
        if variant in {"constructive_pair_accept_anneal", "constructive_pair_accept_anneal32"}:
            set_nested(config, "qdte.accepted_per_iter_schedule", "cosine")
            set_nested(config, "qdte.accepted_per_iter_end", 2)
        if variant == "constructive_pair_accept_anneal32":
            set_nested(config, "qdte.accepted_per_iter", 32)
            set_nested(config, "qdte.accepted_per_iter_start", 32)
    elif variant == "protected_same_row":
        set_nested(config, "qdte.candidate_compiler", "protected_same_row")
        set_nested(config, "qdte.transport_mode", "constructive_pair")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.protected_repair_seed_fraction", 0.5)
        set_nested(config, "qdte.protected_repair_harm_queries", 4)
        set_nested(config, "qdte.protected_repair_restarts_per_seed", 2)
        set_nested(config, "qdte.protected_repair_max_protection_passes", 2)
        set_nested(config, "qdte.protected_repair_require_target_direction", True)
        set_nested(config, "qdte.constructive_pair_partner_limit", 16)
        set_nested(config, "qdte.constructive_pair_harm_query_limit", 16)
    elif variant == "constructive_pair_mixture":
        set_nested(config, "qdte.candidate_compiler", "qdte_mixture")
        set_nested(config, "qdte.transport_mode", "constructive_pair")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.random_candidate_fraction", 0.125)
        set_nested(config, "qdte.candidate_shortfall_policy", "random")
        set_nested(config, "qdte.qdte_mixture_single_fraction", 0.60)
        set_nested(config, "qdte.qdte_mixture_masked_single_fraction", 0.10)
        set_nested(config, "qdte.qdte_mixture_enumerated_fraction", 0.25)
        set_nested(config, "qdte.qdte_mixture_relaxed_masked_fraction", 0.05)
        set_nested(config, "qdte.constructive_pair_partner_limit", 16)
        set_nested(config, "qdte.constructive_pair_harm_query_limit", 16)
    elif variant == "constructive_partner":
        set_nested(config, "qdte.candidate_compiler", "constructive_partner")
        set_nested(config, "qdte.transport_mode", "constructive_pair")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.constructive_partner_seed_fraction", 0.5)
        set_nested(config, "qdte.constructive_partner_harm_queries", 4)
        set_nested(config, "qdte.constructive_partner_partners_per_seed", 1)
        set_nested(config, "qdte.constructive_partner_source_over_sample_factor", 8)
        set_nested(config, "qdte.constructive_pair_partner_limit", 16)
        set_nested(config, "qdte.constructive_pair_harm_query_limit", 16)
    elif variant in {
        "constructive_partner_b2",
        "constructive_partner_b2_accept_anneal",
        "constructive_partner_b2_accept_anneal32",
    }:
        set_nested(config, "qdte.candidate_compiler", "constructive_partner_b2")
        set_nested(config, "qdte.transport_mode", "constructive_pair")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.constructive_partner_seed_fraction", 0.5)
        set_nested(config, "qdte.constructive_partner_harm_queries", 4)
        set_nested(config, "qdte.constructive_partner_partners_per_seed", 1)
        set_nested(config, "qdte.constructive_partner_source_over_sample_factor", 8)
        set_nested(config, "qdte.constructive_partner_side_budget", 128)
        set_nested(config, "qdte.constructive_pair_partner_limit", 16)
        set_nested(config, "qdte.constructive_pair_harm_query_limit", 16)
        if variant in {"constructive_partner_b2_accept_anneal", "constructive_partner_b2_accept_anneal32"}:
            set_nested(config, "qdte.accepted_per_iter_schedule", "cosine")
            set_nested(config, "qdte.accepted_per_iter_end", 2)
        if variant == "constructive_partner_b2_accept_anneal32":
            set_nested(config, "qdte.accepted_per_iter", 32)
            set_nested(config, "qdte.accepted_per_iter_start", 32)
    elif variant in {
        "bounded_best_partner",
        "best_partner_d",
        "bounded_best_partner_cached",
        "best_partner_d_cached",
        "bounded_best_partner_cached_verified",
        "best_partner_d_cached_verified",
        "bounded_best_partner_unique",
        "best_partner_d_unique",
        "bounded_best_partner_source_cached",
        "best_partner_d_source_cached",
        "bounded_best_partner_gpu",
        "bounded_best_partner_gpu_exact",
        "bounded_best_partner_jax_fixed",
        "best_partner_d_gpu",
        "best_partner_d_gpu_exact",
        "bounded_best_partner_fast",
    }:
        set_nested(config, "qdte.candidate_compiler", "bounded_best_partner")
        set_nested(config, "qdte.transport_mode", "constructive_pair")
        set_nested(config, "qdte.transport_prefix_strategy", "best_advantage")
        set_nested(config, "qdte.best_partner_seed_fraction", 0.5)
        set_nested(config, "qdte.best_partner_harm_queries", 4)
        set_nested(config, "qdte.best_partner_partners_per_seed", 1)
        set_nested(config, "qdte.best_partner_source_samples", 64)
        set_nested(config, "qdte.best_partner_repairs_per_source", 2)
        set_nested(config, "qdte.best_partner_side_budget", 128)
        if variant in {"bounded_best_partner_gpu", "best_partner_d_gpu", "bounded_best_partner_fast"}:
            set_nested(config, "qdte.best_partner_delta_backend", "jax_batch")
            set_nested(config, "qdte.best_partner_jax_batch_size", 16384)
            set_nested(config, "qdte.best_partner_seed_batch_size", 32)
        elif variant in {
            "bounded_best_partner_gpu_exact",
            "bounded_best_partner_jax_fixed",
            "best_partner_d_gpu_exact",
        }:
            set_nested(config, "qdte.best_partner_delta_backend", "jax_fixed")
            set_nested(config, "qdte.best_partner_jax_batch_size", 1024)
        elif variant in {"bounded_best_partner_cached", "best_partner_d_cached"}:
            set_nested(config, "qdte.best_partner_delta_backend", "dense_cached")
        elif variant in {"bounded_best_partner_cached_verified", "best_partner_d_cached_verified"}:
            set_nested(config, "qdte.best_partner_delta_backend", "dense_cached")
            set_nested(config, "qdte.best_partner_cached_verify_margin", 1.0e-4)
        elif variant in {"bounded_best_partner_unique", "best_partner_d_unique"}:
            set_nested(config, "qdte.best_partner_delta_backend", "dense_unique")
        elif variant in {"bounded_best_partner_source_cached", "best_partner_d_source_cached"}:
            set_nested(config, "qdte.best_partner_delta_backend", "source_cached")
        else:
            set_nested(config, "qdte.best_partner_delta_backend", "dense_cpu")
        set_nested(config, "qdte.constructive_pair_partner_limit", 16)
        set_nested(config, "qdte.constructive_pair_harm_query_limit", 16)
    else:
        raise ValueError(f"Unknown variant: {requested_variant}")
    if blind_accept:
        set_nested(config, "qdte.transport_mode", "blind_accept")
    config = apply_overrides(config, _drop_output_dir_override(overrides))
    run_qdte(config)


if __name__ == "__main__":
    main()
