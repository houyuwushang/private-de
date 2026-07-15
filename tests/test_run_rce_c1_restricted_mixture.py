from __future__ import annotations

from pathlib import Path

import numpy as np

from qdte.evolution.entropy import ReleasedProductPrior
from scripts.run_rce_c1_restricted_mixture import _component_distributions


ROOT = Path(__file__).resolve().parents[1]


def test_component_distributions_use_full_row_atom_probabilities() -> None:
    prior = ReleasedProductPrior(
        cardinalities=(2, 2),
        probabilities=(
            np.asarray([0.6, 0.4]),
            np.asarray([0.25, 0.75]),
        ),
        public_total=4,
        smoothing=1.0,
    )
    tables = (
        np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32),
        np.asarray([[0, 0], [0, 1], [1, 1], [1, 1]], dtype=np.int32),
        np.asarray([[0, 1], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
    )
    probabilities, support_codes, log_prior, support_hash = _component_distributions(
        tables,
        prior,
    )
    assert probabilities.shape == (3, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert len(support_codes) == len(log_prior) == 4
    assert len(support_hash) == 64
    assert not np.allclose(probabilities[0], probabilities[1])


def test_restricted_mixture_runner_has_no_truth_input_surface() -> None:
    source = (ROOT / "scripts" / "run_rce_c1_restricted_mixture.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "real_encoded.npy",
        "external_inputs",
        "evaluate_external_synthetic",
        "true_answers",
        "official_aim",
    ):
        assert forbidden not in source
