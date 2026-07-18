import numpy as np

from qdte.rce.streamwise import (
    StreamwiseRCEConfidenceSet,
    build_cdwf_streamwise_confidence,
)


def _confidence() -> StreamwiseRCEConfidenceSet:
    return build_cdwf_streamwise_confidence(
        combined_target=np.asarray([1.0, 2.0, 3.0]),
        base_target=np.asarray([1.0, 2.0, 3.0]),
        base_variances=np.asarray([1.0, 1.0, 1.0]),
        refinement_targets=(
            np.asarray([2.0]),
            np.asarray([3.0, 1.0]),
        ),
        refinement_variances=(
            np.asarray([0.25]),
            np.asarray([0.5, 0.5]),
        ),
        refinement_indices=(
            np.asarray([1]),
            np.asarray([2, 0]),
        ),
    )


def test_streamwise_confidence_intersects_all_observation_streams() -> None:
    confidence = _confidence()
    evaluation = confidence.evaluate_answer(np.asarray([1.0, 2.0, 3.0]))
    assert evaluation["inside"]
    assert evaluation["num_streams"] == 3
    assert confidence.confidence_level == 0.95


def test_streamwise_confidence_uses_each_stream_center_not_combined_only() -> None:
    confidence = _confidence()
    answer = np.asarray([1.0, 20.0, 3.0])
    evaluation = confidence.evaluate_answer(answer)
    assert not evaluation["inside"]
    assert not evaluation["streams"]["refinement_00"]["inside"]


def test_streamwise_combined_residual_round_trip_and_serialization() -> None:
    confidence = _confidence()
    combined_residual = np.asarray([0.25, -0.5, 1.0])
    answer = confidence.canonical_answer_from_combined_residual(combined_residual)
    assert np.allclose(answer, np.asarray([0.75, 2.5, 2.0]))
    restored = StreamwiseRCEConfidenceSet.from_public_dict(
        confidence.to_public_dict()
    )
    assert restored.stream_hash == confidence.stream_hash
    assert restored.to_public_dict() == confidence.to_public_dict()
