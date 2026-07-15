from scripts import evaluate_static_ice_wp10a_workload_allocation as evaluator


def _metrics(primary: float, tail: float) -> dict[str, float]:
    return {
        **{name: primary for name in evaluator.PRIMARY_METRICS},
        **{name: tail for name in evaluator.TAIL_METRICS},
    }


def test_wp10a_gate_requires_primary_gain_and_metric_safety() -> None:
    control = _metrics(1.0, 1.0)
    passed = evaluator.apply_gate(_metrics(0.95, 1.10), control)
    assert passed["passed"]
    assert passed["primary_composite_ratio"] == 0.95

    no_gain = evaluator.apply_gate(_metrics(0.98, 1.0), control)
    assert not no_gain["passed"]
    assert not no_gain["checks"]["primary_composite_at_most_0p97"]

    unsafe_tail = evaluator.apply_gate(_metrics(0.90, 1.16), control)
    assert not unsafe_tail["passed"]
    assert not unsafe_tail["checks"]["each_tail_at_most_1p15"]


def test_wp10a_gate_exposes_frozen_thresholds() -> None:
    gate = evaluator.apply_gate(_metrics(0.9, 1.0), _metrics(1.0, 1.0))
    assert gate["thresholds"] == {
        "primary_composite_max": 0.97,
        "primary_metric_max": 1.05,
        "tail_metric_max": 1.15,
    }
