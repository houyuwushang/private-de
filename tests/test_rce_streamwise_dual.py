import numpy as np

from qdte.rce.streamwise import build_cdwf_streamwise_confidence
from qdte.rce.streamwise_dual import StreamwiseRCEDualState


def _dual() -> StreamwiseRCEDualState:
    confidence = build_cdwf_streamwise_confidence(
        combined_target=np.asarray([1.0, 2.0, 3.0]),
        base_target=np.asarray([1.2, 1.8, 3.1]),
        base_variances=np.asarray([1.0, 2.0, 3.0]),
        refinement_targets=(np.asarray([2.2, 2.9]),),
        refinement_variances=(np.asarray([0.25, 0.5]),),
        refinement_indices=(np.asarray([1, 2]),),
    )
    dual = StreamwiseRCEDualState.create(confidence, max_iterations=100)
    dual.ellipsoid_weights[:] = np.asarray([0.7, 1.3])
    dual.tube_positive_by_stream[0][:] = np.asarray([0.1, 0.2, 0.3])
    dual.tube_negative_by_stream[0][:] = np.asarray([0.4, 0.1, 0.2])
    dual.tube_positive_by_stream[1][:] = np.asarray([0.5, 0.2])
    dual.tube_negative_by_stream[1][:] = np.asarray([0.1, 0.6])
    return dual


def test_streamwise_dual_edit_gain_is_exact_finite_difference() -> None:
    dual = _dual()
    residual = np.asarray([0.5, -0.25, 0.75])
    delta = np.asarray([0.2, -0.4, 0.1])
    regularizer_gain = 0.03
    edit_cost = 2.0
    lambda_cost = 0.01
    gain = dual.batch_gain(
        combined_residual=residual,
        canonical_delta_sum=delta,
        regularizer_gain=regularizer_gain,
        edit_cost=edit_cost,
        lambda_cost=lambda_cost,
    )
    before = dual.lagrangian(residual, regularizer=1.4)
    after = dual.lagrangian(
        residual - delta,
        regularizer=1.4 - regularizer_gain,
    )
    assert np.isclose(
        gain,
        before - after - lambda_cost * edit_cost,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_streamwise_dual_update_reads_every_stream() -> None:
    dual = _dual()
    before = dual.ellipsoid_weights.copy()
    record = dual.update(np.asarray([20.0, -20.0, 15.0]))
    assert dual.num_updates == 1
    assert record["num_streams"] == 2
    assert np.any(dual.ellipsoid_weights > before)
    assert record["active_positive_duals"] > 0
