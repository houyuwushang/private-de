from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import jax
import numpy as np

from qdte.evolution.candidates import generate_candidates
from qdte.evolution.entropy import ReleasedProductPrior, RowReferencePrior
from qdte.evolution.precision import OrthogonalInteractionPrecision
from qdte.evolution.scoring import (
    prepare_orthogonal_precision_score_context,
    score_candidates_orthogonal_coefficient_weights_with_quadratic,
)
from qdte.measurement.cdwf import coefficient_block_slices
from qdte.measurement.factorization import HierarchicalInteractionTranscript
from qdte.queries.types import QueryCatalogue
from qdte.queries.workload import WorkloadGroup
from qdte.rce.cdwf_dual import (
    CDWFPressureResult,
    CertifiedRestrictedRCEDual,
    compute_cdwf_block_pressures,
    extract_certified_restricted_dual,
)
from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.forest_prior import ReleasedConfidenceForestPrior
from qdte.rce.relaxed import RestrictedMixtureRCEResult, solve_restricted_mixture_rce
from qdte.schema import TableSchema


CDWF_SHADOW_DICTIONARY_METHOD = "released_directed_table_path_shadow_dictionary_v2"
CDWF_SHADOW_SOLVE_METHOD = "certified_restricted_shadow_rce_v1"


def _readonly_rows(rows: np.ndarray, cardinalities: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.int32).copy()
    if array.ndim != 2 or array.shape[1] != len(cardinalities) or len(array) == 0:
        raise ValueError("CDWF shadow rows must be a non-empty schema-aligned table")
    for attribute, cardinality in enumerate(cardinalities):
        if np.any(array[:, attribute] < 0) or np.any(
            array[:, attribute] >= cardinality
        ):
            raise ValueError("CDWF shadow rows contain an out-of-domain value")
    array.setflags(write=False)
    return array


def sample_released_product_prior(
    prior: ReleasedProductPrior,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(n_rows) <= 0:
        raise ValueError("CDWF prior sample size must be positive")
    return np.column_stack(
        [
            rng.choice(cardinality, size=int(n_rows), p=probabilities)
            for cardinality, probabilities in zip(
                prior.cardinalities,
                prior.probabilities,
                strict=True,
            )
        ]
    ).astype(np.int32)


def sample_released_forest_prior(
    prior: ReleasedConfidenceForestPrior,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = int(n_rows)
    if count <= 0:
        raise ValueError("CDWF forest sample size must be positive")
    dimension = prior.dimension
    adjacency: list[list[tuple[int, Any]]] = [[] for _ in range(dimension)]
    edge_by_pair = {edge.pair: edge for edge in prior.edges}
    for edge in prior.edges:
        left, right = edge.pair
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))
    rows = np.zeros((count, dimension), dtype=np.int32)
    visited: set[int] = set()
    for root in range(dimension):
        if root in visited:
            continue
        rows[:, root] = rng.choice(
            prior.cardinalities[root],
            size=count,
            p=prior.probabilities[root],
        )
        visited.add(root)
        queue = [root]
        while queue:
            parent = queue.pop(0)
            for child, edge in sorted(adjacency[parent], key=lambda item: item[0]):
                if child in visited:
                    continue
                left, right = edge.pair
                table = edge_by_pair[(left, right)].table
                for parent_value in range(prior.cardinalities[parent]):
                    row_indices = np.flatnonzero(rows[:, parent] == parent_value)
                    if len(row_indices) == 0:
                        continue
                    if parent == left:
                        probabilities = table[parent_value, :] / prior.probabilities[
                            parent
                        ][parent_value]
                    else:
                        probabilities = table[:, parent_value] / prior.probabilities[
                            parent
                        ][parent_value]
                    probabilities = np.maximum(probabilities, 0.0)
                    probabilities /= float(np.sum(probabilities))
                    rows[row_indices, child] = rng.choice(
                        prior.cardinalities[child],
                        size=len(row_indices),
                        p=probabilities,
                    )
                visited.add(child)
                queue.append(child)
    return rows


@dataclass(frozen=True)
class CDWFShadowDictionary:
    names: tuple[str, ...]
    tables: tuple[np.ndarray, ...]
    cardinalities: tuple[int, ...]
    public_seed: int
    coefficient_answers: tuple[np.ndarray, ...] | None = None
    strategy_signature: str | None = None
    path_diagnostics: dict[str, Any] | None = None
    method: str = CDWF_SHADOW_DICTIONARY_METHOD

    def __post_init__(self) -> None:
        if self.method != CDWF_SHADOW_DICTIONARY_METHOD:
            raise ValueError(f"Unsupported CDWF shadow dictionary {self.method!r}")
        names = tuple(str(name) for name in self.names)
        required_prefix = (
            "base_initial",
            "base_product_sample",
            "base_ccf_sample",
        )
        if names[: len(required_prefix)] != required_prefix:
            raise ValueError("CDWF shadow dictionary is missing its three released anchors")
        if len(names) != len(set(names)):
            raise ValueError("CDWF shadow dictionary names must be unique")
        tables = tuple(
            _readonly_rows(table, self.cardinalities) for table in self.tables
        )
        if len(tables) != len(names) or len({len(table) for table in tables}) != 1:
            raise ValueError("CDWF shadow tables must share a public row count")
        coefficient_answers = self.coefficient_answers
        if coefficient_answers is not None:
            coefficient_answers = tuple(
                np.asarray(values, dtype=np.float64).copy()
                for values in coefficient_answers
            )
            if len(coefficient_answers) != len(tables):
                raise ValueError("CDWF coefficient answers must match the shadow tables")
            dimensions = {values.shape for values in coefficient_answers}
            if len(dimensions) != 1 or next(iter(dimensions))[0] <= 0:
                raise ValueError("CDWF coefficient answers must share a positive dimension")
            if any(not np.all(np.isfinite(values)) for values in coefficient_answers):
                raise ValueError("CDWF coefficient answers must be finite")
            for values in coefficient_answers:
                values.setflags(write=False)
            if not self.strategy_signature:
                raise ValueError("Cached CDWF coefficient answers require a strategy signature")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "coefficient_answers", coefficient_answers)

    @classmethod
    def create(
        cls,
        *,
        initial_rows: np.ndarray,
        product_prior: ReleasedProductPrior,
        ccf_prior: ReleasedConfidenceForestPrior,
        public_seed: int,
    ) -> CDWFShadowDictionary:
        initial = np.asarray(initial_rows, dtype=np.int32)
        if product_prior.cardinalities != ccf_prior.cardinalities:
            raise ValueError("CDWF shadow priors must use the same public schema")
        seed_sequence = np.random.SeedSequence(
            [int(public_seed), 0x43445746, 0x53484457]
        )
        product_seed, forest_seed = seed_sequence.spawn(2)
        return cls(
            names=(
                "base_initial",
                "base_product_sample",
                "base_ccf_sample",
            ),
            tables=(
                initial,
                sample_released_product_prior(
                    product_prior,
                    len(initial),
                    np.random.default_rng(product_seed),
                ),
                sample_released_forest_prior(
                    ccf_prior,
                    len(initial),
                    np.random.default_rng(forest_seed),
                ),
            ),
            cardinalities=product_prior.cardinalities,
            public_seed=int(public_seed),
        )

    @property
    def dictionary_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.method.encode("ascii"))
        digest.update(json.dumps(self.names, separators=(",", ":")).encode("ascii"))
        digest.update(np.asarray(self.cardinalities, dtype="<i8").tobytes())
        for table in self.tables:
            digest.update(np.asarray(table, dtype="<i4").tobytes(order="C"))
        if self.strategy_signature is not None:
            digest.update(self.strategy_signature.encode("ascii"))
        return digest.hexdigest()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scope": "released_only_frozen_directed_table_path_hull",
            "global_pricing_certificate": False,
            "names": list(self.names),
            "num_tables": len(self.tables),
            "num_rows_per_table": len(self.tables[0]),
            "cardinalities": list(self.cardinalities),
            "public_seed": self.public_seed,
            "dictionary_hash": self.dictionary_hash,
            "strategy_signature": self.strategy_signature,
            "path_diagnostics": self.path_diagnostics,
        }

    def component_distributions(
        self,
        prior: RowReferencePrior,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tuple(prior.cardinalities) != self.cardinalities:
            raise ValueError("CDWF shadow prior schema differs from the dictionary")
        all_rows = np.concatenate(self.tables, axis=0)
        all_codes = prior.encode_rows(all_rows)
        support_codes, first_indices, inverse = np.unique(
            all_codes,
            return_index=True,
            return_inverse=True,
        )
        probabilities = np.zeros((len(self.tables), len(support_codes)), dtype=np.float64)
        offset = 0
        for index, table in enumerate(self.tables):
            local = inverse[offset : offset + len(table)]
            probabilities[index] = np.bincount(
                local,
                minlength=len(support_codes),
            ) / float(len(table))
            offset += len(table)
        support_rows = all_rows[first_indices]
        log_prior = prior.log_probability_rows(support_rows)
        return probabilities, support_rows, log_prior


def _strategy_signature(transcript: HierarchicalInteractionTranscript) -> str:
    payload = {
        "cardinalities": list(transcript.strategy.cardinalities),
        "pairs": [list(pair) for pair in transcript.strategy.pairs],
        "blocks": [block.to_dict() for block in transcript.strategy.blocks],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _coefficient_answer(
    precision: OrthogonalInteractionPrecision,
    rows: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    total = np.zeros(precision.coefficient_dimension, dtype=np.float64)
    records = np.asarray(rows, dtype=np.int32)
    for start in range(0, len(records), int(batch_size)):
        total += np.sum(
            precision.row_features(records[start : start + int(batch_size)]),
            axis=0,
            dtype=np.float64,
        )
    return total


def _active_query_ids_from_released_gradient(
    precision: OrthogonalInteractionPrecision,
    weighted_coefficient_residual: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the released coefficient gradient back to query coordinates."""

    query_gradient = precision.coefficient_adjoint(
        np.asarray(weighted_coefficient_residual, dtype=np.float64)
    )
    finite = np.flatnonzero(np.isfinite(query_gradient))
    count = min(max(0, int(limit)), len(finite))
    if count == 0:
        return np.empty(0, dtype=np.int32), query_gradient
    order = np.lexsort((finite, -np.abs(query_gradient[finite])))
    return finite[order[:count]].astype(np.int32), query_gradient


def _select_exact_shadow_batch(
    *,
    precision: OrthogonalInteractionPrecision,
    row_ids: np.ndarray,
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    weighted_residual: np.ndarray,
    inverse_variances: np.ndarray,
    gpu_scores: np.ndarray,
    accepted_cap: int,
    delta_chunk_size: int = 128,
) -> tuple[list[int], np.ndarray, float]:
    """Select a nonconflicting batch and audit its exact aggregate advantage."""

    order = np.argsort(-np.asarray(gpu_scores, dtype=np.float64), kind="stable")
    positive_order = order[np.asarray(gpu_scores)[order] > 0.0]
    aggregate = np.zeros(precision.coefficient_dimension, dtype=np.float64)
    selected_candidates: list[int] = []
    selected_rows: set[int] = set()
    for chunk_start in range(0, len(positive_order), int(delta_chunk_size)):
        chunk_indices = positive_order[
            chunk_start : chunk_start + int(delta_chunk_size)
        ]
        deltas = precision.row_feature_deltas(
            old_rows[chunk_indices],
            new_rows[chunk_indices],
        )
        for local_index, candidate_index in enumerate(chunk_indices.tolist()):
            row_id = int(row_ids[candidate_index])
            if row_id in selected_rows:
                continue
            delta = deltas[local_index]
            incremental = float(
                delta @ (weighted_residual - inverse_variances * aggregate)
                - 0.5 * np.sum(delta * delta * inverse_variances)
            )
            if incremental <= 1.0e-12:
                continue
            aggregate += delta
            selected_candidates.append(int(candidate_index))
            selected_rows.add(row_id)
            if len(selected_candidates) >= int(accepted_cap):
                break
        if len(selected_candidates) >= int(accepted_cap):
            break
    exact_advantage = float(
        aggregate @ weighted_residual
        - 0.5 * np.sum(aggregate * aggregate * inverse_variances)
    )
    if selected_candidates and exact_advantage <= 0.0:
        raise RuntimeError("CDWF shadow accepted a nonpositive aggregate edit batch")
    return selected_candidates, aggregate, exact_advantage


def build_released_directed_shadow_dictionary(
    dictionary: CDWFShadowDictionary,
    *,
    schema: TableSchema,
    qcat: QueryCatalogue,
    workload_groups: Sequence[WorkloadGroup],
    transcript: HierarchicalInteractionTranscript,
    released_target: np.ndarray,
    max_rounds: int = 512,
    candidates_per_round: int = 1_024,
    accepted_per_round: int = 64,
    num_active_targets: int = 64,
    random_candidate_fraction: float = 0.05,
    source_over_sample_factor: int = 32,
    feature_batch_size: int = 4_096,
) -> CDWFShadowDictionary:
    """Freeze a released-only QDTE path for the restricted shadow oracle.

    The path is built once from the base transcript. Every accepted batch uses
    the exact finite-difference decrease of the frozen weighted quadratic
    objective. Later C3 rounds may only optimize mixtures of these frozen
    row-realizable tables; they do not extend the dictionary.
    """

    if int(max_rounds) <= 0 or int(candidates_per_round) <= 0:
        raise ValueError("CDWF directed shadow search sizes must be positive")
    if int(accepted_per_round) <= 0:
        raise ValueError("CDWF directed shadow accepted_per_round must be positive")
    if tuple(schema.cardinalities) != dictionary.cardinalities:
        raise ValueError("CDWF directed shadow schema differs from the dictionary")
    if int(num_active_targets) <= 0:
        raise ValueError("CDWF directed shadow num_active_targets must be positive")
    if not 0.0 <= float(random_candidate_fraction) <= 1.0:
        raise ValueError("CDWF shadow random candidate fraction must be in [0, 1]")
    if int(source_over_sample_factor) <= 0:
        raise ValueError("CDWF shadow source over-sample factor must be positive")
    precision = OrthogonalInteractionPrecision(
        qcat,
        list(workload_groups),
        transcript,
    )
    target = np.asarray(released_target, dtype=np.float64)
    if target.shape != (qcat.m,) or not np.all(np.isfinite(target)):
        raise ValueError("CDWF directed shadow target must match the query catalogue")
    signature = _strategy_signature(transcript)
    coefficient_answers = [
        _coefficient_answer(
            precision,
            table,
            batch_size=int(feature_batch_size),
        )
        for table in dictionary.tables
    ]
    target_coefficients = precision.coefficient_coordinates(target)
    variances = precision.coefficient_variances
    inverse_variances = np.divide(
        1.0,
        variances,
        out=np.zeros_like(variances),
        where=variances > 0.0,
    )
    confidence = RCEConfidenceSet.from_diagonal_variances(
        variances,
        alpha_l2=0.025,
        alpha_linf=0.025,
    )
    anchor_residuals = [
        target_coefficients - answer for answer in coefficient_answers
    ]
    anchor_evaluations = [
        confidence.evaluate(anchor_residual) for anchor_residual in anchor_residuals
    ]
    start_index = min(
        range(len(anchor_evaluations)),
        key=lambda index: (anchor_evaluations[index].slack, index),
    )
    current = np.asarray(dictionary.tables[start_index], dtype=np.int32).copy()
    current_answer = coefficient_answers[start_index].copy()
    residual = anchor_residuals[start_index].copy()
    score_context = prepare_orthogonal_precision_score_context(precision)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(dictionary.public_seed), 0x43445746, 0x44495245]
        )
    )
    names = list(dictionary.names)
    tables = [np.asarray(table, dtype=np.int32).copy() for table in dictionary.tables]
    diagnostics: list[dict[str, Any]] = []
    snapshot_rounds = set(range(15, int(max_rounds), 16))
    snapshot_rounds.update({0, 1, 3, int(max_rounds) - 1})
    initial_evaluation = confidence.evaluate(residual)
    diagnostics.append(
        {
            "round": -1,
            "objective": float(0.5 * np.sum(residual * residual * inverse_variances)),
            "confidence_slack": float(initial_evaluation.slack),
            "inside_confidence": bool(initial_evaluation.inside),
            "accepted_edits": 0,
            "exact_batch_advantage": 0.0,
        }
    )
    for round_index in range(int(max_rounds)):
        weighted_residual = residual * inverse_variances
        active_query_ids, query_gradient = _active_query_ids_from_released_gradient(
            precision,
            weighted_residual,
            limit=int(num_active_targets),
        )
        if len(active_query_ids) == 0:
            break
        candidates_per_target = max(
            1,
            int(np.ceil(int(candidates_per_round) / len(active_query_ids))),
        )
        candidates = generate_candidates(
            current,
            qcat,
            schema,
            active_query_ids,
            query_gradient,
            {
                "qdte": {
                    "candidate_compiler": "single_query",
                    "candidates_per_target": candidates_per_target,
                    "total_candidates_per_iter": int(candidates_per_round),
                    "random_candidate_fraction": float(random_candidate_fraction),
                    "source_over_sample_factor": int(source_over_sample_factor),
                    "numerical_distance_gamma": 0.0,
                    "lambda_cost": 0.0,
                    "candidate_shortfall_policy": "random",
                }
            },
            rng,
        )
        if candidates.size == 0:
            break
        scores, _ = score_candidates_orthogonal_coefficient_weights_with_quadratic(
            candidates,
            weighted_residual,
            precision,
            0.0,
            chunk_size=int(candidates_per_round),
            context=score_context,
        )
        selected_candidates, aggregate, exact_batch_advantage = (
            _select_exact_shadow_batch(
                precision=precision,
                row_ids=candidates.row_ids,
                old_rows=candidates.old_rows,
                new_rows=candidates.new_rows,
                weighted_residual=weighted_residual,
                inverse_variances=inverse_variances,
                gpu_scores=scores,
                accepted_cap=int(accepted_per_round),
            )
        )
        if not selected_candidates:
            break
        for candidate_index in selected_candidates:
            current[int(candidates.row_ids[candidate_index])] = candidates.new_rows[
                candidate_index
            ]
        current_answer += aggregate
        residual -= aggregate
        evaluation = confidence.evaluate(residual)
        objective = float(0.5 * np.sum(residual * residual * inverse_variances))
        diagnostics.append(
            {
                "round": round_index,
                "objective": objective,
                "confidence_slack": float(evaluation.slack),
                "inside_confidence": bool(evaluation.inside),
                "accepted_edits": len(selected_candidates),
                "exact_batch_advantage": exact_batch_advantage,
                "active_queries": len(active_query_ids),
                "generated_candidates": candidates.size,
                "directed_candidates": int(
                    round(candidates.diagnostics.get("directed_candidates", 0.0))
                ),
                "random_candidates": int(
                    round(candidates.diagnostics.get("random_candidates", 0.0))
                ),
            }
        )
        if round_index in snapshot_rounds or evaluation.inside:
            names.append(f"directed_fit_round_{round_index + 1:03d}")
            tables.append(current.copy())
            coefficient_answers.append(current_answer.copy())
        if evaluation.inside:
            break
    final = diagnostics[-1]
    return CDWFShadowDictionary(
        names=tuple(names),
        tables=tuple(tables),
        cardinalities=dictionary.cardinalities,
        public_seed=dictionary.public_seed,
        coefficient_answers=tuple(coefficient_answers),
        strategy_signature=signature,
        path_diagnostics={
            "method": "released_qdte_base_compiler_exact_quadratic_table_path_v2",
            "max_rounds": int(max_rounds),
            "candidates_per_round": int(candidates_per_round),
            "accepted_per_round": int(accepted_per_round),
            "num_active_targets": int(num_active_targets),
            "candidate_compiler": "single_query",
            "random_candidate_fraction": float(random_candidate_fraction),
            "source_over_sample_factor": int(source_over_sample_factor),
            "active_query_score": "absolute_query_adjoint_gradient",
            "path_start_anchor": dictionary.names[start_index],
            "anchor_confidence_slack": {
                name: float(evaluation.slack)
                for name, evaluation in zip(
                    dictionary.names,
                    anchor_evaluations,
                    strict=True,
                )
            },
            "candidate_scoring_backend": str(jax.default_backend()),
            "candidate_scoring_devices": [str(device) for device in jax.devices()],
            "candidate_scoring_precision": "float32_rank_float64_batch_audit",
            "rounds_executed": len(diagnostics) - 1,
            "entered_confidence_set": bool(final["inside_confidence"]),
            "initial_objective": float(diagnostics[0]["objective"]),
            "final_objective": float(final["objective"]),
            "initial_confidence_slack": float(diagnostics[0]["confidence_slack"]),
            "final_confidence_slack": float(final["confidence_slack"]),
            "snapshots": diagnostics,
        },
    )


@dataclass(frozen=True)
class CDWFShadowSolve:
    result: RestrictedMixtureRCEResult
    dual: CertifiedRestrictedRCEDual
    pressure: CDWFPressureResult | None
    confidence: RCEConfidenceSet
    component_answers: np.ndarray
    restricted_dictionary: CDWFShadowDictionary
    fallback_required: bool
    failure_reasons: tuple[str, ...]
    method: str = CDWF_SHADOW_SOLVE_METHOD

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "restricted_dictionary": self.restricted_dictionary.to_public_dict(),
            "result": self.result.to_dict(),
            "dual": self.dual.to_public_dict(),
            "pressure": self.pressure.to_public_dict() if self.pressure else None,
            "fallback_required": self.fallback_required,
            "failure_reasons": list(self.failure_reasons),
        }


def solve_cdwf_shadow_rce(
    dictionary: CDWFShadowDictionary,
    *,
    qcat: QueryCatalogue,
    workload_groups: Sequence[WorkloadGroup],
    transcript: HierarchicalInteractionTranscript,
    released_target: np.ndarray,
    prior: RowReferencePrior,
    rho_by_block: dict[str, float],
    answer_batch_size: int = 8192,
    max_iterations: int = 5_000,
) -> CDWFShadowSolve:
    precision = OrthogonalInteractionPrecision(
        qcat,
        list(workload_groups),
        transcript,
    )
    confidence = RCEConfidenceSet.from_diagonal_variances(
        precision.coefficient_variances,
        alpha_l2=0.025,
        alpha_linf=0.025,
    )
    target = np.asarray(released_target, dtype=np.float64)
    if target.shape != (qcat.m,) or not np.all(np.isfinite(target)):
        raise ValueError("CDWF shadow target must match the query catalogue")
    if dictionary.coefficient_answers is not None:
        if dictionary.strategy_signature != _strategy_signature(transcript):
            raise ValueError("CDWF shadow coefficient cache does not match the strategy")
        component_answers = np.stack(dictionary.coefficient_answers, axis=0)
    else:
        component_answers = np.stack(
            [
                _coefficient_answer(
                    precision,
                    table,
                    batch_size=int(answer_batch_size),
                )
                for table in dictionary.tables
            ],
            axis=0,
        )
    coefficient_target = precision.coefficient_coordinates(target)
    coefficient_residuals = coefficient_target.reshape(1, -1) - component_answers
    component_probabilities, _, log_prior = dictionary.component_distributions(prior)
    result = solve_restricted_mixture_rce(
        component_probabilities,
        coefficient_residuals,
        log_prior,
        confidence,
        component_names=dictionary.names,
        max_iterations=int(max_iterations),
    )
    dual = extract_certified_restricted_dual(result, confidence)
    reasons = list(dual.failure_reasons)
    pressure: CDWFPressureResult | None = None
    if dual.eligible:
        pressure = compute_cdwf_block_pressures(
            residual=result.residual,
            confidence=confidence,
            dual=dual,
            block_slices=coefficient_block_slices(transcript.strategy),
            rho_by_block=rho_by_block,
        )
        if not pressure.autodiff_audit_passed:
            reasons.append("pressure_autodiff_audit_failed")
    return CDWFShadowSolve(
        result=result,
        dual=dual,
        pressure=pressure,
        confidence=confidence,
        component_answers=component_answers,
        restricted_dictionary=dictionary,
        fallback_required=bool(reasons),
        failure_reasons=tuple(reasons),
    )


__all__ = [
    "CDWF_SHADOW_DICTIONARY_METHOD",
    "CDWF_SHADOW_SOLVE_METHOD",
    "CDWFShadowDictionary",
    "CDWFShadowSolve",
    "build_released_directed_shadow_dictionary",
    "sample_released_forest_prior",
    "sample_released_product_prior",
    "solve_cdwf_shadow_rce",
]
