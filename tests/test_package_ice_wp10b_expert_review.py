from pathlib import Path

from scripts import package_ice_wp10b_expert_review as package


def test_build_wp10b_review_preserves_failed_public_gate(tmp_path: Path) -> None:
    output = tmp_path / "review" / "wp10b" / "repo"
    manifest = package.build_wp10b_review(package.ROOT, output, force=False)

    assert manifest["package_id"] == package.PACKAGE_ID
    assert not package.verify_wp10b_review(output)
    assert not list(output.rglob("*.npy"))
    assert not list(output.rglob("*.npz"))
    for dataset in package.PUBLIC_GATE_DATASETS:
        for name in package.PUBLIC_GATE_INPUT_NAMES:
            assert (output / "external_inputs" / f"{dataset}_sage_strong" / name).is_file()
    gate = package.read_json(
        output / "outputs/static_ice_wp10b_gcea_public_gate_20260715/public_gate.json"
    )
    assert gate["generation_authorized"] is False
    assert gate["decision"] == "q1_a_freeze_ice_binary_profile"


def test_wp10b_review_readme_points_to_frozen_result() -> None:
    text = package._readme()
    assert "SAGE_QDTE_ICE_WP10B_GCEA_PUBLIC_GATE_RESULT_20260715.md" in text
    assert "generation_authorized=false" in text
