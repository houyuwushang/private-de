from __future__ import annotations

from pathlib import Path

from scripts import package_ice_wp10a_expert_review as package


def test_sanitize_string_uses_repository_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "private-de"
    external = tmp_path / "baseline" / "private-de" / "external_inputs"
    assert package.sanitize_string(str(source / "docs" / "result.md"), source) == "docs/result.md"
    assert (
        package.sanitize_string(str(external / "adult" / "schema.json"), source)
        == "external_inputs/adult/schema.json"
    )
    assert package.sanitize_string(str(external), source) == "external_inputs"
    assert package.sanitize_string("/mnt/archive/raw.json", source) == "external_artifacts/raw.json"


def test_build_review_package_is_self_contained_and_sanitized(tmp_path: Path) -> None:
    output = tmp_path / "review" / "snapshot" / "repo"
    manifest = package.build_review_package(package.ROOT, output, force=False)

    assert manifest["package_id"] == package.PACKAGE_ID
    assert not package.verify_review_package(output)
    assert (output / "outputs/static_ice_wp9_figures_20260715/wp9_static_ice_vs_aim_gate.png").is_file()
    assert (output / "outputs/static_ice_wp10a_workload_l2_eval_20260715/gate_summary.json").is_file()
    assert not list(output.rglob("offline_true_answer_cache*"))
    assert not list(output.rglob("*.npy"))
    assert not list(output.rglob("*.npz"))

    wp10a = (output / package.DOCUMENT_PATHS[-1]).read_text(encoding="utf-8")
    assert "/home/" not in wp10a
    assert "../outputs/static_ice_wp9_figures_20260715/wp9_static_ice_vs_aim_gate.png" in wp10a
