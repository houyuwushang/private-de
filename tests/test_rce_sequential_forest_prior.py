import numpy as np

from qdte.rce.sequential_forest_prior import (
    CCFInteractionStream,
    solve_sequential_ccf_pair,
)


def _stream(name: str, center: float, variance: float, round_index: int):
    return CCFInteractionStream(
        name=name,
        released_interaction_rate=np.asarray([[center]], dtype=np.float64),
        interaction_covariance_rate=np.asarray([[variance]], dtype=np.float64),
        confidence_radius=3.841458820694124,
        alpha_struct=0.025 if round_index < 0 else 0.0125,
        round_index=round_index,
    )


def test_sequential_ccf_returns_product_only_when_all_streams_contain_it() -> None:
    result = solve_sequential_ccf_pair(
        (0, 1),
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
        (
            _stream("base", 0.01, 0.01, -1),
            _stream("round0", -0.01, 0.01, 0),
        ),
    )
    assert result.product_in_confidence
    assert result.status == "product_in_confidence"
    assert result.weight_lower_bound == 0.0


def test_sequential_ccf_certifies_common_dependence_across_streams() -> None:
    result = solve_sequential_ccf_pair(
        (0, 1),
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
        (
            _stream("base", 0.20, 1.0e-3, -1),
            _stream("round0", 0.18, 1.0e-3, 0),
        ),
    )
    assert result.eligible
    assert result.pair_table is not None
    assert result.weight_lower_bound > 0.0
    assert result.maximum_ellipsoid_violation <= 1.0e-8
    assert result.stationarity_residual <= 1.0e-7
    assert np.allclose(result.pair_table.sum(axis=0), 0.5, atol=1.0e-10)
    assert np.allclose(result.pair_table.sum(axis=1), 0.5, atol=1.0e-10)


def test_sequential_ccf_fail_closes_on_certified_inconsistent_streams() -> None:
    result = solve_sequential_ccf_pair(
        (0, 1),
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
        (
            _stream("base", 0.20, 1.0e-4, -1),
            _stream("round0", -0.20, 1.0e-4, 0),
        ),
    )
    assert result.status == "anchor_interaction_inconsistent"
    assert not result.eligible
    assert result.pair_table is None
    assert result.feasibility_lower > 1.0
