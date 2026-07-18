from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from scipy.stats import chi2


@dataclass(frozen=True)
class ChiSquareDiscrepancyStop:
    """Stop once a quadratic Gaussian discrepancy enters its confidence set.

    QDTE quadratic objectives use ``L = 0.5 * r.T @ W @ r``. When ``W`` is the
    exact precision of an independent Gaussian transcript, ``2 * L`` is the
    corresponding squared whitened discrepancy and is compared with a
    chi-square quantile using the public effective rank of ``W``.
    """

    alpha: float
    effective_rank: int
    squared_discrepancy_threshold: float

    @classmethod
    def create(cls, *, alpha: float, effective_rank: int) -> ChiSquareDiscrepancyStop:
        alpha_value = float(alpha)
        rank_value = int(effective_rank)
        if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        if rank_value <= 0:
            raise ValueError("effective_rank must be positive")
        threshold = float(chi2.ppf(1.0 - alpha_value, rank_value))
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("chi-square threshold must be finite and positive")
        return cls(
            alpha=alpha_value,
            effective_rank=rank_value,
            squared_discrepancy_threshold=threshold,
        )

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.alpha

    @property
    def objective_threshold(self) -> float:
        return 0.5 * self.squared_discrepancy_threshold

    def reached(self, objective: float) -> bool:
        value = float(objective)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("objective must be finite and nonnegative")
        tolerance = 1.0e-12 * max(1.0, self.objective_threshold)
        return value <= self.objective_threshold + tolerance

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "chi_square",
            "alpha": self.alpha,
            "confidence_level": self.confidence_level,
            "effective_rank": self.effective_rank,
            "squared_discrepancy_threshold": self.squared_discrepancy_threshold,
            "objective_threshold": self.objective_threshold,
        }
