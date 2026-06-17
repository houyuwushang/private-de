from __future__ import annotations

import numpy as np
import pytest

from qdte.measurement.consistency import (
    project_consistent_targets,
    project_local_table_feasible_lsq,
    project_local_table_feasible_jax,
    project_query_space_feasible_lsq,
    project_query_space_lsq,
)
from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryBuilder


def test_consistency_projection_enforces_known_total_for_complete_partition() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()

    result = project_consistent_targets(
        np.asarray([8.0, 10.0], dtype=np.float32),
        qcat,
        np.asarray([2], dtype=np.int32),
        total=10,
        variances=np.ones(2, dtype=np.float32),
    )

    assert np.all(result.projected >= 0.0)
    assert np.isclose(float(result.projected.sum()), 10.0, atol=1.0e-5)
    assert result.diagnostics["known_total_count"] == 10


def test_query_space_lsq_enforces_known_total_without_nonnegative_clipping() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()

    result = project_query_space_lsq(
        np.asarray([-10.0, 25.0], dtype=np.float32),
        qcat,
        np.asarray([2], dtype=np.int32),
        total=10,
        variances=np.ones(2, dtype=np.float32),
    )

    assert result.diagnostics["method"] == "query_space_lsq"
    assert np.isclose(float(result.projected.sum()), 10.0, atol=1.0e-5)
    assert float(result.projected[0]) < 0.0


def test_query_space_feasible_lsq_enforces_nonnegative_known_total() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()

    result = project_query_space_feasible_lsq(
        np.asarray([-10.0, 25.0], dtype=np.float32),
        qcat,
        np.asarray([2], dtype=np.int32),
        total=10,
        variances=np.ones(2, dtype=np.float32),
    )

    assert result.diagnostics["method"] == "query_space_feasible_lsq"
    assert result.diagnostics["solver_success"] is True
    assert np.all(result.projected >= -1.0e-6)
    assert np.all(result.projected <= 10.0 + 1.0e-6)
    assert np.isclose(float(result.projected.sum()), 10.0, atol=1.0e-5)
    assert np.allclose(result.projected, np.asarray([0.0, 10.0], dtype=np.float32), atol=1.0e-5)


def test_query_space_lsq_moves_high_variance_answers_more_for_total_constraint() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()

    result = project_query_space_lsq(
        np.asarray([8.0, 10.0], dtype=np.float32),
        qcat,
        np.asarray([2], dtype=np.int32),
        total=10,
        variances=np.asarray([1.0, 9.0], dtype=np.float32),
    )

    assert np.allclose(result.projected, np.asarray([7.2, 2.8], dtype=np.float32), atol=1.0e-5)
    assert np.isclose(float(result.projected.sum()), 10.0, atol=1.0e-5)


def test_query_space_feasible_lsq_matches_unconstrained_solution_when_bounds_inactive() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()

    result = project_query_space_feasible_lsq(
        np.asarray([8.0, 10.0], dtype=np.float32),
        qcat,
        np.asarray([2], dtype=np.int32),
        total=10,
        variances=np.asarray([1.0, 9.0], dtype=np.float32),
    )

    assert np.allclose(result.projected, np.asarray([7.2, 2.8], dtype=np.float32), atol=1.0e-5)
    assert np.isclose(float(result.projected.sum()), 10.0, atol=1.0e-5)


def test_query_space_lsq_aligns_lower_and_higher_complete_marginals() -> None:
    builder = QueryBuilder(max_terms=2)
    a0 = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    a1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    ab: dict[tuple[int, int], int] = {}
    for a in range(2):
        for b in range(2):
            ab[(a, b)] = len(builder.names)
            builder.add(
                [(0, OP_EQ, a, a, a), (1, OP_EQ, b, b, b)],
                f"a={a}&b={b}",
                "twoway:0:1",
                "twoway",
            )
    qcat = builder.build()
    noisy = np.asarray([9.0, 1.0, 3.0, 4.0, 2.0, 1.0], dtype=np.float32)

    result = project_query_space_lsq(
        noisy,
        qcat,
        np.asarray([2, 2], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
    )
    z = result.projected

    assert np.isclose(float(z[a0] + z[a1]), 10.0, atol=1.0e-5)
    assert np.isclose(float(sum(z[ab[(a, b)]] for a in range(2) for b in range(2))), 10.0, atol=1.0e-5)
    assert np.isclose(float(z[a0]), float(z[ab[(0, 0)]] + z[ab[(0, 1)]]), atol=1.0e-5)
    assert np.isclose(float(z[a1]), float(z[ab[(1, 0)]] + z[ab[(1, 1)]]), atol=1.0e-5)


def test_query_space_feasible_lsq_aligns_lower_and_higher_complete_marginals() -> None:
    builder = QueryBuilder(max_terms=2)
    a0 = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    a1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    ab: dict[tuple[int, int], int] = {}
    for a in range(2):
        for b in range(2):
            ab[(a, b)] = len(builder.names)
            builder.add(
                [(0, OP_EQ, a, a, a), (1, OP_EQ, b, b, b)],
                f"a={a}&b={b}",
                "twoway:0:1",
                "twoway",
            )
    qcat = builder.build()
    noisy = np.asarray([9.0, 1.0, 3.0, 4.0, 2.0, 1.0], dtype=np.float32)

    result = project_query_space_feasible_lsq(
        noisy,
        qcat,
        np.asarray([2, 2], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
    )
    z = result.projected

    assert np.all(z >= -1.0e-6)
    assert np.all(z <= 10.0 + 1.0e-6)
    assert np.isclose(float(z[a0] + z[a1]), 10.0, atol=1.0e-5)
    assert np.isclose(float(sum(z[ab[(a, b)]] for a in range(2) for b in range(2))), 10.0, atol=1.0e-5)
    assert np.isclose(float(z[a0]), float(z[ab[(0, 0)]] + z[ab[(0, 1)]]), atol=1.0e-5)
    assert np.isclose(float(z[a1]), float(z[ab[(1, 0)]] + z[ab[(1, 1)]]), atol=1.0e-5)


def test_local_table_feasible_lsq_aligns_lower_and_higher_complete_marginals() -> None:
    builder = QueryBuilder(max_terms=2)
    a0 = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    a1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    ab: dict[tuple[int, int], int] = {}
    for a in range(2):
        for b in range(2):
            ab[(a, b)] = len(builder.names)
            builder.add(
                [(0, OP_EQ, a, a, a), (1, OP_EQ, b, b, b)],
                f"a={a}&b={b}",
                "twoway:0:1",
                "twoway",
            )
    qcat = builder.build()
    noisy = np.asarray([9.0, 1.0, 3.0, 4.0, 2.0, 1.0], dtype=np.float32)

    result = project_local_table_feasible_lsq(
        noisy,
        qcat,
        np.asarray([2, 2], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
    )
    z = result.projected

    assert result.diagnostics["method"] == "local_table_feasible_lsq"
    assert result.diagnostics["solver_success"] is True
    assert np.all(z >= -1.0e-6)
    assert np.all(z <= 10.0 + 1.0e-6)
    assert np.isclose(float(z[a0] + z[a1]), 10.0, atol=1.0e-5)
    assert np.isclose(float(sum(z[ab[(a, b)]] for a in range(2) for b in range(2))), 10.0, atol=1.0e-5)
    assert np.isclose(float(z[a0]), float(z[ab[(0, 0)]] + z[ab[(0, 1)]]), atol=1.0e-5)
    assert np.isclose(float(z[a1]), float(z[ab[(1, 0)]] + z[ab[(1, 1)]]), atol=1.0e-5)


def test_local_table_feasible_lsq_supports_latent_mixed_scope_without_complete_cells() -> None:
    builder = QueryBuilder(max_terms=2)
    a0 = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    a1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    mixed = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0), (1, OP_LE, 1, 0, 1)], "a=0&b<=1", "mixed:0:1", "mixed")
    qcat = builder.build()
    noisy = np.asarray([8.0, 8.0, 12.0], dtype=np.float32)

    result = project_local_table_feasible_lsq(
        noisy,
        qcat,
        np.asarray([2, 3], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
    )
    z = result.projected

    assert result.diagnostics["num_scopes"] == 2
    assert result.diagnostics["num_shared_marginal_constraints"] > 0
    assert np.all(z >= -1.0e-6)
    assert np.all(z <= 10.0 + 1.0e-6)
    assert np.isclose(float(z[a0] + z[a1]), 10.0, atol=1.0e-5)
    assert float(z[mixed]) <= float(z[a0]) + 1.0e-5


def test_local_table_feasible_jax_enforces_scope_simplex() -> None:
    builder = QueryBuilder(max_terms=1)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    qcat = builder.build()

    result = project_local_table_feasible_jax(
        np.asarray([-10.0, 25.0], dtype=np.float32),
        qcat,
        np.asarray([2], dtype=np.int32),
        total=10,
        variances=np.ones(2, dtype=np.float32),
        jax_iterations=100,
    )

    assert result.diagnostics["method"] == "local_table_feasible_jax"
    assert result.diagnostics["solver_success"] is True
    assert np.all(result.projected >= -1.0e-5)
    assert np.all(result.projected <= 10.0 + 1.0e-5)
    assert np.isclose(float(result.projected.sum()), 10.0, atol=1.0e-4)


def test_query_space_lsq_aligns_prefix_and_range_to_complete_cells() -> None:
    builder = QueryBuilder(max_terms=1)
    cells = []
    for x in range(4):
        cells.append(len(builder.names))
        builder.add([(0, OP_EQ, x, x, x)], f"x={x}", "oneway:0", "oneway")
    prefix_2 = len(builder.names)
    builder.add([(0, OP_LE, 2, 0, 2)], "x<=2", "prefix:0", "prefix")
    range_12 = len(builder.names)
    builder.add([(0, OP_RANGE, 1, 1, 2)], "x[1,2]", "range:0", "range")
    qcat = builder.build()
    noisy = np.asarray([1.0, 2.0, 3.0, 4.0, 20.0, 9.0], dtype=np.float32)

    result = project_query_space_lsq(
        noisy,
        qcat,
        np.asarray([4], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
    )
    z = result.projected

    assert np.isclose(float(sum(z[qid] for qid in cells)), 10.0, atol=1.0e-5)
    assert np.isclose(float(z[prefix_2]), float(z[cells[0]] + z[cells[1]] + z[cells[2]]), atol=1.0e-5)
    assert np.isclose(float(z[range_12]), float(z[cells[1]] + z[cells[2]]), atol=1.0e-5)


def test_consistency_projection_reconciles_three_way_and_lower_marginals() -> None:
    builder = QueryBuilder(max_terms=3)
    qids: dict[tuple[int, ...], int] = {}

    for a in range(2):
        qids[(a,)] = len(builder.names)
        builder.add([(0, OP_EQ, a, a, a)], f"a={a}", "oneway:0", "oneway")
    for a in range(2):
        for b in range(2):
            qids[(a, b)] = len(builder.names)
            builder.add(
                [(0, OP_EQ, a, a, a), (1, OP_EQ, b, b, b)],
                f"a={a}&b={b}",
                "twoway:0:1",
                "twoway",
            )
    for a in range(2):
        for b in range(2):
            for c in range(2):
                qids[(a, b, c)] = len(builder.names)
                builder.add(
                    [(0, OP_EQ, a, a, a), (1, OP_EQ, b, b, b), (2, OP_EQ, c, c, c)],
                    f"a={a}&b={b}&c={c}",
                    "threeway:0:1:2",
                    "mixed",
                )
    qcat = builder.build()
    noisy = np.asarray(
        [
            9.0,
            5.0,
            7.0,
            1.0,
            4.0,
            3.0,
            8.0,
            2.0,
            6.0,
            2.0,
            1.0,
            5.0,
            2.0,
            7.0,
        ],
        dtype=np.float32,
    )

    result = project_consistent_targets(
        noisy,
        qcat,
        np.asarray([2, 2, 2], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
        max_iterations=200,
        tolerance=1.0e-8,
    )
    z = result.projected

    assert np.isclose(float(z[[qids[(0,)], qids[(1,)]]].sum()), 10.0, atol=1.0e-4)
    for a in range(2):
        assert np.isclose(
            float(sum(z[qids[(a, b)]] for b in range(2))),
            float(z[qids[(a,)]]),
            atol=1.0e-3,
        )
        for b in range(2):
            assert np.isclose(
                float(sum(z[qids[(a, b, c)]] for c in range(2))),
                float(z[qids[(a, b)]]),
                atol=1.0e-3,
            )


def test_consistency_projection_handles_prefix_range_and_mixed_queries() -> None:
    builder = QueryBuilder(max_terms=2)
    c0 = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0)], "c=0", "oneway:0", "oneway")
    c1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "c=1", "oneway:0", "oneway")
    prefix_qids: list[int] = []
    for threshold in range(4):
        prefix_qids.append(len(builder.names))
        builder.add([(1, OP_LE, threshold, 0, threshold)], f"x<={threshold}", "prefix:1", "prefix")
    range_12 = len(builder.names)
    builder.add([(1, OP_RANGE, 1, 1, 2)], "x[1,2]", "range:1", "range")
    mixed = len(builder.names)
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_LE, 2, 0, 2)],
        "c=1&x<=2",
        "mixed:0:1",
        "mixed",
    )
    for c in range(2):
        for x in range(4):
            builder.add(
                [(0, OP_EQ, c, c, c), (1, OP_EQ, x, x, x)],
                f"c={c}&x={x}",
                "cells:0:1",
                "mixed",
            )
    qcat = builder.build()
    noisy = np.linspace(1.0, 20.0, qcat.m, dtype=np.float32)

    result = project_consistent_targets(
        noisy,
        qcat,
        np.asarray([2, 4], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
        max_iterations=200,
        tolerance=1.0e-8,
    )
    z = result.projected

    assert np.isclose(float(z[c0] + z[c1]), 10.0, atol=1.0e-4)
    assert np.all(np.diff(z[prefix_qids]) >= -1.0e-5)
    assert np.isclose(float(z[prefix_qids[-1]]), 10.0, atol=1.0e-3)
    assert np.isclose(float(z[range_12]), float(z[prefix_qids[2]] - z[prefix_qids[0]]), atol=1.0e-3)
    assert z[mixed] <= z[c1] + 1.0e-3
    assert z[mixed] <= z[prefix_qids[2]] + 1.0e-3


def test_consistency_projection_supports_ge_and_four_dimensional_conjunction() -> None:
    builder = QueryBuilder(max_terms=4)
    a1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    ge_b = len(builder.names)
    builder.add([(1, OP_GE, 1, 1, 2)], "b>=1", "prefix:1", "prefix")
    q4 = len(builder.names)
    builder.add(
        [
            (0, OP_EQ, 1, 1, 1),
            (1, OP_GE, 1, 1, 2),
            (2, OP_RANGE, 0, 0, 1),
            (3, OP_LE, 1, 0, 1),
        ],
        "a=1&b>=1&c[0,1]&d<=1",
        "fourway",
        "mixed",
    )
    qcat = builder.build()

    result = project_consistent_targets(
        np.asarray([8.0, 9.0, 7.0], dtype=np.float32),
        qcat,
        np.asarray([2, 3, 2, 2], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
        max_iterations=100,
    )
    z = result.projected

    assert np.all(z >= -1.0e-6)
    assert z[q4] <= z[a1] + 1.0e-4
    assert z[q4] <= z[ge_b] + 1.0e-4
    assert result.diagnostics["max_scope_cells_observed"] == 24


def test_consistency_projection_fails_fast_when_scope_is_too_large() -> None:
    builder = QueryBuilder(max_terms=3)
    builder.add(
        [(0, OP_EQ, 1, 1, 1), (1, OP_EQ, 2, 2, 2), (2, OP_EQ, 3, 3, 3)],
        "large",
        "large",
        "mixed",
    )
    qcat = builder.build()

    with pytest.raises(ValueError, match="max_scope_cells"):
        project_consistent_targets(
            np.asarray([1.0], dtype=np.float32),
            qcat,
            np.asarray([10, 10, 10], dtype=np.int32),
            total=100,
            max_scope_cells=999,
        )


def test_consistency_projection_handles_halfspace_with_shared_oneway_marginal() -> None:
    builder = QueryBuilder(max_terms=2)
    a0 = len(builder.names)
    builder.add([(0, OP_EQ, 0, 0, 0)], "a=0", "oneway:0", "oneway")
    a1 = len(builder.names)
    builder.add([(0, OP_EQ, 1, 1, 1)], "a=1", "oneway:0", "oneway")
    h = len(builder.names)
    builder.add_halfspace([(0, 1.0), (1, 1.0)], threshold=2.0, name="a+x<=2", group="halfspace")
    after_first = len(builder.names)
    builder.add_halfspace([(0, 1.0), (1, 1.0)], threshold=2.0, name="a+x<=2-again", group="dup")
    qcat = builder.build()
    assert len(builder.names) == after_first  # duplicate halfspace is removed by QueryBuilder

    result = project_consistent_targets(
        np.asarray([9.0, 8.0, 12.0], dtype=np.float32)[: qcat.m],
        qcat,
        np.asarray([2, 4], dtype=np.int32),
        total=10,
        variances=np.ones(qcat.m, dtype=np.float32),
        max_iterations=100,
    )
    z = result.projected

    assert np.isclose(float(z[a0] + z[a1]), 10.0, atol=1.0e-4)
    assert 0.0 <= z[h] <= 10.0
    assert result.diagnostics["num_scopes"] == 2
