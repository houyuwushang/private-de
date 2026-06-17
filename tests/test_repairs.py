from __future__ import annotations

import numpy as np
import pytest

from qdte.evolution.candidates import generate_candidates, repair_enter, repair_exit
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryBuilder
from qdte.schema import ColumnSchema, TableSchema


def _check_repair(term: tuple[int, int, int, int, int], unsat: np.ndarray, sat: np.ndarray) -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([term], "q", "g", "f")
    qcat = builder.build()
    cards = np.asarray([5], dtype=np.int32)
    rng = np.random.default_rng(0)
    entered = repair_enter(unsat.copy(), qcat, 0, cards, rng)
    exited = repair_exit(sat.copy(), qcat, 0, cards, rng)
    assert bool(qcat.eval_query_np(entered[None, :], 0)[0])
    assert not bool(qcat.eval_query_np(exited[None, :], 0)[0])
    assert 0 <= entered[0] < cards[0]
    assert 0 <= exited[0] < cards[0]


def test_repair_eq() -> None:
    _check_repair((0, OP_EQ, 2, 2, 2), np.asarray([0], dtype=np.int32), np.asarray([2], dtype=np.int32))


def test_repair_le() -> None:
    _check_repair((0, OP_LE, 2, 0, 2), np.asarray([4], dtype=np.int32), np.asarray([1], dtype=np.int32))


def test_repair_ge() -> None:
    _check_repair((0, OP_GE, 2, 2, 4), np.asarray([0], dtype=np.int32), np.asarray([3], dtype=np.int32))


def test_repair_range() -> None:
    _check_repair((0, OP_RANGE, 1, 1, 3), np.asarray([4], dtype=np.int32), np.asarray([2], dtype=np.int32))


def _assert_valid_codes(row: np.ndarray, cards: np.ndarray) -> None:
    assert np.all(row >= 0)
    assert np.all(row < cards)


def _assert_direct_repair_contract(qcat, qid: int, unsat: np.ndarray, sat: np.ndarray, cards: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    entered = repair_enter(unsat.copy(), qcat, qid, cards, rng)
    exited = repair_exit(sat.copy(), qcat, qid, cards, rng)

    assert bool(qcat.eval_query_np(entered[None, :], qid)[0])
    assert not bool(qcat.eval_query_np(exited[None, :], qid)[0])
    _assert_valid_codes(entered, cards)
    _assert_valid_codes(exited, cards)


def test_directed_repair_contract_for_kway_conjunction() -> None:
    builder = QueryBuilder(max_terms=4)
    builder.add(
        [
            (0, OP_EQ, 2, 2, 2),
            (1, OP_LE, 1, 0, 1),
            (2, OP_GE, 3, 3, 4),
            (3, OP_RANGE, 1, 1, 3),
        ],
        "a=2&b<=1&c>=3&d[1,3]",
        "mixed",
        "mixed",
    )
    qcat = builder.build()
    cards = np.asarray([5, 5, 5, 5], dtype=np.int32)

    _assert_direct_repair_contract(
        qcat,
        0,
        unsat=np.asarray([0, 4, 0, 4], dtype=np.int32),
        sat=np.asarray([2, 1, 3, 2], dtype=np.int32),
        cards=cards,
    )


def test_directed_repair_contract_for_halfspace() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add_halfspace([(0, 1.0), (1, 1.0)], threshold=2.0, name="a+b<=2", group="halfspace")
    qcat = builder.build()
    cards = np.asarray([5, 5], dtype=np.int32)

    _assert_direct_repair_contract(
        qcat,
        0,
        unsat=np.asarray([4, 4], dtype=np.int32),
        sat=np.asarray([1, 1], dtype=np.int32),
        cards=cards,
    )


def test_generate_candidates_respects_enter_exit_contract_for_active_queries() -> None:
    builder = QueryBuilder(max_terms=3)
    builder.add(
        [(0, OP_EQ, 2, 2, 2), (1, OP_LE, 1, 0, 1), (2, OP_GE, 3, 3, 4)],
        "a=2&b<=1&c>=3",
        "mixed",
        "mixed",
    )
    builder.add_halfspace([(1, 1.0), (2, 1.0)], threshold=2.0, name="b+c<=2", group="halfspace")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 5),
            ColumnSchema("b", "numerical_binned", 5),
            ColumnSchema("c", "numerical_binned", 5),
            ColumnSchema("d", "categorical", 5),
        ]
    )
    X_syn = np.asarray(
        [
            [0, 4, 0, 0],
            [1, 3, 1, 1],
            [3, 4, 4, 2],
            [2, 0, 4, 0],
            [0, 0, 0, 0],
            [2, 1, 1, 0],
            [4, 0, 2, 1],
            [1, 1, 0, 3],
        ],
        dtype=np.int32,
    )
    residual = np.asarray([5.0, -5.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidates_per_target": 8,
            "total_candidates_per_iter": 24,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 16,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(7),
    )

    assert candidates.size > 0
    directed_qids = set(int(qid) for qid in candidates.target_query_ids.tolist() if int(qid) >= 0)
    assert directed_qids == {0, 1}
    for idx in range(candidates.size):
        qid = int(candidates.target_query_ids[idx])
        if qid < 0:
            continue
        old_sat = bool(qcat.eval_query_np(candidates.old_rows[idx : idx + 1], qid)[0])
        new_sat = bool(qcat.eval_query_np(candidates.new_rows[idx : idx + 1], qid)[0])
        if residual[qid] > 0:
            assert not old_sat
            assert new_sat
        else:
            assert old_sat
            assert not new_sat


def test_paired_query_compiler_moves_homogeneous_source_to_underfit_destination() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.zeros((16, 2), dtype=np.int32)
    residual = np.asarray([-8.0, 8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "paired_query",
            "paired_candidate_fraction": 1.0,
            "paired_try_break_source": True,
            "paired_source_over_sample_factor": 4,
            "paired_candidates_per_pair": 8,
            "paired_pair_rounds": 2,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(11),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["paired_candidates"] == 8.0
    assert np.all(candidates.repair_type == 3)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert bool(qcat.eval_query_np(old, 0)[0])
        assert not bool(qcat.eval_query_np(old, 1)[0])
        assert not bool(qcat.eval_query_np(new, 0)[0])
        assert bool(qcat.eval_query_np(new, 1)[0])


def test_masked_paired_query_compiler_uses_partial_destination_terms() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1), (2, OP_EQ, 1, 1, 1)], "b=1&c=1", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
            ColumnSchema("c", "categorical", 2),
        ]
    )
    X_syn = np.zeros((16, 3), dtype=np.int32)
    residual = np.asarray([-8.0, 8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "masked_paired_query",
            "paired_candidate_fraction": 1.0,
            "paired_try_break_source": False,
            "paired_source_over_sample_factor": 4,
            "paired_candidates_per_pair": 8,
            "paired_pair_rounds": 2,
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(13),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["paired_candidates"] == 8.0
    assert candidates.diagnostics["masked_paired_candidates"] == 8.0
    assert np.all(candidates.repair_type == 4)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert bool(qcat.eval_query_np(old, 0)[0])
        assert not bool(qcat.eval_query_np(old, 1)[0])
        assert not bool(qcat.eval_query_np(new, 1)[0])
        assert int(new[0, 1] == 1) + int(new[0, 2] == 1) == 1


def test_masked_exit_query_compiler_preserves_underfit_mask_while_exiting_source() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 0, 0, 0), (1, OP_EQ, 0, 0, 0)], "a=0&b=0", "mixed", "mixed")
    builder.add([(2, OP_EQ, 1, 1, 1)], "c=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
            ColumnSchema("c", "categorical", 2),
        ]
    )
    X_syn = np.tile(np.asarray([[0, 0, 1]], dtype=np.int32), (16, 1))
    residual = np.asarray([-8.0, 8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "masked_exit_query",
            "paired_candidate_fraction": 1.0,
            "paired_source_over_sample_factor": 4,
            "paired_candidates_per_pair": 8,
            "paired_pair_rounds": 2,
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(17),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["paired_candidates"] == 8.0
    assert candidates.diagnostics["masked_exit_candidates"] == 8.0
    assert np.all(candidates.repair_type == 5)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert bool(qcat.eval_query_np(old, 0)[0])
        assert bool(qcat.eval_query_np(old, 1)[0])
        assert not bool(qcat.eval_query_np(new, 0)[0])
        assert bool(qcat.eval_query_np(new, 1)[0])
        assert int(new[0, 2]) == 1


def test_masked_single_query_enter_uses_near_miss_sources() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)], "a=1&b=1", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1],
            ],
            dtype=np.int32,
        ),
        (8, 1),
    )
    residual = np.asarray([8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "masked_single_query",
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 8,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(31),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["masked_single_query_candidates"] == 8.0
    assert np.all(candidates.repair_type == 9)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert not bool(qcat.eval_query_np(old, 0)[0])
        assert bool(qcat.eval_query_np(new, 0)[0])
        assert int(old[0, 0] == 1) + int(old[0, 1] == 1) == 1
        assert int((old != new).sum()) == 1


def test_masked_single_query_exit_breaks_masked_term_from_full_source() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 0, 0, 0), (1, OP_EQ, 0, 0, 0)], "a=0&b=0", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(np.asarray([[0, 0]], dtype=np.int32), (16, 1))
    residual = np.asarray([-8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "masked_single_query",
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(37),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["masked_single_query_candidates"] == 8.0
    assert np.all(candidates.repair_type == 9)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert bool(qcat.eval_query_np(old, 0)[0])
        assert not bool(qcat.eval_query_np(new, 0)[0])
        assert int((old != new).sum()) == 1


def test_relaxed_masked_single_query_enter_can_use_far_miss_sources() -> None:
    builder = QueryBuilder(max_terms=3)
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1), (2, OP_EQ, 1, 1, 1)],
        "a=1&b=1&c=1",
        "mixed",
        "mixed",
    )
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
            ColumnSchema("c", "categorical", 2),
        ]
    )
    X_syn = np.zeros((24, 3), dtype=np.int32)
    residual = np.asarray([8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "relaxed_masked_single_query",
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(39),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["relaxed_masked_single_query_candidates"] == 8.0
    assert np.all(candidates.repair_type == 14)
    assert np.all(candidates.target_query_ids == 0)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert not bool(qcat.eval_query_np(old, 0)[0])
        assert not bool(qcat.eval_query_np(new, 0)[0])
        assert int((old != new).sum()) == 1
        assert int(new.sum()) == 1


def test_directed_exit_only_compiler_generates_negative_residual_exits() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(np.asarray([[0, 0]], dtype=np.int32), (16, 1))
    residual = np.asarray([-8.0, 8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "directed_exit_only",
            "candidates_per_target": 4,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(19),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["directed_exit_only_candidates"] == 8.0
    assert candidates.diagnostics["random_candidates"] == 0.0
    assert np.all(candidates.repair_type == 6)
    assert np.all(candidates.target_query_ids == 0)
    for idx in range(candidates.size):
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert bool(qcat.eval_query_np(old, 0)[0])
        assert not bool(qcat.eval_query_np(new, 0)[0])


def test_masked_exit_only_compiler_exits_negative_residual_mask() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 0, 0, 0), (1, OP_EQ, 0, 0, 0)], "a=0&b=0", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(np.asarray([[0, 0]], dtype=np.int32), (16, 1))
    residual = np.asarray([-8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "masked_exit_only",
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "candidates_per_target": 4,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(29),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["masked_exit_only_candidates"] == 8.0
    assert np.all(candidates.repair_type == 8)
    assert np.all(candidates.target_query_ids == 0)
    assert np.all((candidates.old_rows == np.asarray([0, 0], dtype=np.int32)).all(axis=1))
    assert np.all((candidates.old_rows != candidates.new_rows).sum(axis=1) == 1)
    for idx in range(candidates.size):
        assert not bool(qcat.eval_query_np(candidates.new_rows[idx : idx + 1], 0)[0])


def test_random_source_directed_exit_compiler_uses_exit_repair_without_source_filter() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(columns=[ColumnSchema("a", "categorical", 3)])
    X_syn = np.tile(np.asarray([[0]], dtype=np.int32), (16, 1))
    residual = np.asarray([-8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "random_source_directed_exit",
            "candidates_per_target": 4,
            "total_candidates_per_iter": 8,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(23),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 8
    assert candidates.diagnostics["random_source_directed_exit_candidates"] == 8.0
    assert candidates.diagnostics["source_filter_attempts"] == 0.0
    assert candidates.diagnostics["random_candidates"] == 0.0
    assert np.all(candidates.repair_type == 7)
    for idx in range(candidates.size):
        new = candidates.new_rows[idx : idx + 1]
        assert not bool(qcat.eval_query_np(new, 0)[0])


@pytest.mark.parametrize(
    ("compiler", "diagnostic", "repair_type"),
    [
        ("residual_weighted_mutation", "residual_weighted_mutation_candidates", 10),
        ("enumerated_local", "enumerated_local_candidates", 11),
        ("residual_value_mutation", "residual_value_mutation_candidates", 13),
    ],
)
def test_broad_proposal_compilers_generate_one_step_candidates(
    compiler: str,
    diagnostic: str,
    repair_type: int,
) -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 3),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [0, 0],
                [1, 1],
                [2, 0],
                [0, 2],
            ],
            dtype=np.int32,
        ),
        (8, 1),
    )
    residual = np.asarray([8.0, -4.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": compiler,
            "candidates_per_target": 8,
            "total_candidates_per_iter": 16,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(41),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size == 16
    assert candidates.diagnostics[diagnostic] == 16.0
    assert candidates.diagnostics["random_candidates"] == 0.0
    assert np.all(candidates.repair_type == repair_type)
    assert np.all((candidates.old_rows != candidates.new_rows).sum(axis=1) == 1)
    assert np.all(candidates.target_query_ids >= 0)


def test_soft_single_query_compiler_keeps_source_filter_soft() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(columns=[ColumnSchema("a", "categorical", 3)])
    X_syn = np.tile(np.asarray([[0], [1], [2]], dtype=np.int32), (12, 1))
    residual = np.asarray([8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "soft_single_query",
            "candidates_per_target": 8,
            "total_candidates_per_iter": 16,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "soft_source_match_weight": 2.0,
            "soft_source_mismatch_weight": 1.0,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(43),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 16
    assert candidates.diagnostics["soft_single_query_candidates"] == 16.0
    assert candidates.diagnostics["random_candidates"] == 0.0
    assert np.all(candidates.repair_type == 12)
    assert np.all(candidates.target_query_ids == 0)


def test_proposal_mixture_combines_random_and_directed_proposals() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 3),
            ColumnSchema("b", "categorical", 3),
        ]
    )
    X_syn = np.tile(np.asarray([[0, 0], [1, 1], [2, 0], [0, 2]], dtype=np.int32), (8, 1))
    residual = np.asarray([8.0, -4.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "proposal_mixture",
            "candidates_per_target": 8,
            "total_candidates_per_iter": 20,
            "random_candidate_fraction": 0.0,
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(47),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size == 20
    assert candidates.diagnostics["proposal_mixture_candidates"] == 20.0
    assert candidates.diagnostics["random_candidates"] > 0.0
    assert candidates.diagnostics["directed_candidates"] > 0.0
    assert set(candidates.repair_type.tolist()).issubset({0, 10, 11, 12, 13})


def test_explicit_candidate_budgets_separate_directed_and_planned_random() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    qcat = builder.build()
    schema = TableSchema(columns=[ColumnSchema("a", "categorical", 3)])
    X_syn = np.tile(np.asarray([[0], [2], [0], [2]], dtype=np.int32), (8, 1))
    residual = np.asarray([8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "single_query",
            "candidates_per_target": 7,
            "total_candidates_per_iter": 10,
            "directed_candidate_count": 7,
            "random_candidate_count": 3,
            "candidate_shortfall_policy": "none",
            "source_over_sample_factor": 4,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(53),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 10
    assert candidates.diagnostics["directed_candidate_budget"] == 7.0
    assert candidates.diagnostics["random_candidate_budget"] == 3.0
    assert candidates.diagnostics["directed_candidates"] == 7.0
    assert candidates.diagnostics["random_candidates"] == 3.0
    assert candidates.diagnostics["planned_random_candidates"] == 3.0
    assert candidates.diagnostics["fallback_random_candidates"] == 0.0
    assert candidates.diagnostics["directed_candidate_shortfall"] == 0.0


def test_qdte_mixture_allocates_single_masked_enumerated_and_relaxed_budgets() -> None:
    builder = QueryBuilder(max_terms=3)
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1), (2, OP_EQ, 1, 1, 1)],
        "a=1&b=1&c=1",
        "mixed",
        "mixed",
    )
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
            ColumnSchema("c", "categorical", 2),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.int32,
        ),
        (16, 1),
    )
    residual = np.asarray([8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "qdte_mixture",
            "candidates_per_target": 3,
            "total_candidates_per_iter": 16,
            "directed_candidate_count": 12,
            "random_candidate_count": 4,
            "candidate_shortfall_policy": "none",
            "qdte_mixture_single_fraction": 0.25,
            "qdte_mixture_masked_single_fraction": 0.25,
            "qdte_mixture_enumerated_fraction": 0.25,
            "qdte_mixture_relaxed_masked_fraction": 0.25,
            "mask_min_terms": 1,
            "mask_max_terms": 1,
            "source_over_sample_factor": 16,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(59),
        inv_variance=np.ones(1, dtype=np.float32),
    )

    assert candidates.size == 16
    assert candidates.diagnostics["qdte_mixture_candidates"] == 12.0
    assert candidates.diagnostics["single_directed_candidates"] == 3.0
    assert candidates.diagnostics["masked_single_query_candidates"] == 3.0
    assert candidates.diagnostics["enumerated_local_candidates"] == 3.0
    assert candidates.diagnostics["relaxed_masked_single_query_candidates"] == 3.0
    assert candidates.diagnostics["planned_random_candidates"] == 4.0
    assert candidates.diagnostics["fallback_random_candidates"] == 0.0
    assert set(candidates.repair_type.tolist()) == {0, 1, 9, 11, 14}


def test_constructive_partner_synthesizes_exit_for_seed_harmed_query() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)], "a=1&b=1", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=np.int32,
        ),
        (8, 1),
    )
    residual = np.asarray([8.0, -8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "constructive_partner",
            "candidates_per_target": 4,
            "total_candidates_per_iter": 12,
            "directed_candidate_count": 12,
            "random_candidate_count": 0,
            "candidate_shortfall_policy": "none",
            "constructive_partner_seed_fraction": 0.5,
            "constructive_partner_harm_queries": 2,
            "constructive_partner_partners_per_seed": 1,
            "constructive_partner_source_over_sample_factor": 16,
            "source_over_sample_factor": 16,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(61),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    partner_idx = np.flatnonzero(candidates.repair_type == 15)
    assert len(partner_idx) > 0
    assert candidates.diagnostics["constructive_partner_candidates"] == float(len(partner_idx))
    assert candidates.diagnostics["constructive_partner_seed_candidates"] > 0.0
    for idx in partner_idx.tolist():
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert candidates.target_query_ids[idx] == 1
        assert bool(qcat.eval_query_np(old, 1)[0])
        assert not bool(qcat.eval_query_np(new, 1)[0])


def test_constructive_partner_b2_preserves_seed_pool_and_attaches_partners() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)], "a=1&b=1", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=np.int32,
        ),
        (8, 1),
    )
    residual = np.asarray([8.0, -8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "constructive_partner_b2",
            "candidates_per_target": 4,
            "total_candidates_per_iter": 12,
            "directed_candidate_count": 12,
            "random_candidate_count": 0,
            "candidate_shortfall_policy": "none",
            "constructive_partner_seed_fraction": 0.5,
            "constructive_partner_harm_queries": 2,
            "constructive_partner_partners_per_seed": 1,
            "constructive_partner_source_over_sample_factor": 16,
            "constructive_partner_side_budget": 6,
            "source_over_sample_factor": 16,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(61),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    assert candidates.size > 8
    assert candidates.diagnostics["requested_candidates"] == 12.0
    assert candidates.diagnostics["constructive_attached_partner_candidates"] > 0.0
    assert candidates.diagnostics["constructive_attached_pair_units"] > 0.0
    assert candidates.attached_pair_indices is not None
    assert len(candidates.attached_pair_indices) > 0
    for seed_idx, partner_idx in candidates.attached_pair_indices.tolist():
        assert candidates.repair_type[int(seed_idx)] in {1, 2}
        assert candidates.repair_type[int(partner_idx)] == 16


@pytest.mark.parametrize(
    "best_partner_delta_backend",
    ["dense_cpu", "dense_unique", "dense_cached", "jax_batch", "jax_fixed"],
)
def test_bounded_best_partner_scores_and_attaches_best_partner(best_partner_delta_backend: str) -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)], "a=1&b=1", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=np.int32,
        ),
        (8, 1),
    )
    residual = np.asarray([8.0, -8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "bounded_best_partner",
            "candidates_per_target": 4,
            "total_candidates_per_iter": 12,
            "directed_candidate_count": 12,
            "random_candidate_count": 0,
            "candidate_shortfall_policy": "none",
            "best_partner_harm_queries": 2,
            "best_partner_partners_per_seed": 1,
            "best_partner_source_samples": 64,
            "best_partner_repairs_per_source": 16,
            "best_partner_side_budget": 8,
            "best_partner_delta_backend": best_partner_delta_backend,
            "best_partner_jax_batch_size": 128,
            "best_partner_seed_batch_size": 4,
            "best_partner_min_pair_advantage": 0.0,
            "lambda_cost": 0.0,
            "source_over_sample_factor": 16,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(61),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    partner_idx = np.flatnonzero(candidates.repair_type == 18)
    assert len(partner_idx) > 0
    assert candidates.diagnostics["best_partner_candidates"] == float(len(partner_idx))
    assert candidates.diagnostics["best_partner_pair_units"] > 0.0
    assert candidates.diagnostics["best_partner_pairs_evaluated"] > 0.0
    assert candidates.diagnostics["best_partner_positive_pairs"] > 0.0
    assert candidates.attached_pair_indices is not None
    assert len(candidates.attached_pair_indices) > 0
    for seed_idx, partner_idx_raw in candidates.attached_pair_indices.tolist():
        partner = int(partner_idx_raw)
        if candidates.repair_type[partner] != 18:
            continue
        assert candidates.repair_type[int(seed_idx)] in {1, 2}
        old = candidates.old_rows[partner : partner + 1]
        new = candidates.new_rows[partner : partner + 1]
        assert bool(qcat.eval_query_np(old, 1)[0])
        assert not bool(qcat.eval_query_np(new, 1)[0])
        break
    else:
        raise AssertionError("expected at least one bounded best-partner attached pair")


def test_protected_same_row_preserves_harmed_query_while_entering_target() -> None:
    builder = QueryBuilder(max_terms=2)
    builder.add([(1, OP_EQ, 1, 1, 1)], "b=1", "oneway", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 1, 1, 1)], "a=1&b=1", "mixed", "mixed")
    qcat = builder.build()
    schema = TableSchema(
        columns=[
            ColumnSchema("a", "categorical", 2),
            ColumnSchema("b", "categorical", 2),
        ]
    )
    X_syn = np.tile(
        np.asarray(
            [
                [1, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.int32,
        ),
        (8, 1),
    )
    residual = np.asarray([8.0, -8.0], dtype=np.float32)
    config = {
        "qdte": {
            "candidate_compiler": "protected_same_row",
            "candidates_per_target": 8,
            "total_candidates_per_iter": 24,
            "directed_candidate_count": 24,
            "random_candidate_count": 0,
            "candidate_shortfall_policy": "none",
            "protected_repair_seed_fraction": 0.5,
            "protected_repair_harm_queries": 2,
            "protected_repair_restarts_per_seed": 16,
            "protected_repair_max_protection_passes": 4,
            "protected_repair_delta_backend": "dense_cpu",
            "source_over_sample_factor": 16,
            "numerical_distance_gamma": 0.0,
        }
    }

    candidates = generate_candidates(
        X_syn,
        qcat,
        schema,
        np.asarray([0, 1], dtype=np.int32),
        residual,
        config,
        np.random.default_rng(71),
        inv_variance=np.ones(2, dtype=np.float32),
    )

    protected_idx = np.flatnonzero(candidates.repair_type == 17)
    assert len(protected_idx) > 0
    assert candidates.diagnostics["protected_same_row_candidates"] == float(len(protected_idx))
    assert candidates.diagnostics["protected_repair_seed_candidates"] > 0.0
    assert candidates.diagnostics["protected_repair_protection_successes"] > 0.0
    for idx in protected_idx.tolist():
        if candidates.target_query_ids[idx] != 0:
            continue
        old = candidates.old_rows[idx : idx + 1]
        new = candidates.new_rows[idx : idx + 1]
        assert not bool(qcat.eval_query_np(old, 0)[0])
        assert bool(qcat.eval_query_np(new, 0)[0])
        assert bool(qcat.eval_query_np(old, 1)[0]) == bool(qcat.eval_query_np(new, 1)[0])
        break
    else:
        raise AssertionError("expected at least one protected target-enter candidate")
