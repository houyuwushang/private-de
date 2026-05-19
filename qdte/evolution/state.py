from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QDTEState:
    X_syn: np.ndarray
    answer_syn: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    variance: np.ndarray
    inv_variance: np.ndarray
    sigma: np.ndarray
    debt: np.ndarray
    iteration: int
