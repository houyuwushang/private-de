from __future__ import annotations

import numpy as np

from qdte.evolution.candidates import repair_enter, repair_exit
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryBuilder


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
