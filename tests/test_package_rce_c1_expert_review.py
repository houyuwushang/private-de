from pathlib import Path

from scripts import package_rce_c1_expert_review as package


def test_build_rce_review_preserves_negative_result_without_raw_data(tmp_path: Path) -> None:
    output = tmp_path / "review" / "rce-c1" / "repo"
    manifest = package.build_rce_review(package.ROOT, output, force=False)

    assert manifest["package_id"] == package.PACKAGE_ID
    assert not package.verify_rce_review(output)
    assert not list(output.rglob("*.npy"))
    assert not list(output.rglob("*.npz"))
    evaluation = package.read_json(
        output / "outputs/sage_qdte_rce_c1_v3_eval_20260715/summary.json"
    )
    assert evaluation["classification"] == {
        "rce_improves_point_target_control": False,
        "adult_beats_aim_aggregate": False,
        "adult_beats_aim_at_both_epsilons": False,
    }
    postseal = package.read_json(
        output / "outputs/sage_qdte_rce_c1_v3_postseal_20260715/summary.json"
    )
    assert postseal["confidence"]["truth"]["inside"] == 12
    restricted = package.read_json(
        output
        / "outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/summary.json"
    )
    assert restricted["classification"] == {
        "beats_rce_v1_overall": True,
        "beats_wp9_overall": False,
        "product_kl_prefers_wp9_on_average": False,
    }


def test_rce_review_readme_points_to_decision_and_evidence() -> None:
    text = package._readme()
    assert "SAGE_QDTE_RCE_C1_RESULT_AND_NEXT_EXPERT_QUESTIONS_20260715.md" in text
    assert "sage_qdte_rce_c1_v3_eval_20260715/summary.json" in text
    assert "sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/summary.json" in text
    assert "excludes" in text
