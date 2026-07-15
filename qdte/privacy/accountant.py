from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def zcdp_epsilon(rho: float, delta: float) -> float:
    if rho <= 0:
        return 0.0
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def bounded_range_rho(epsilon: float) -> float:
    """Exact zCDP conversion for an epsilon-bounded-range mechanism."""
    eps = float(epsilon)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if eps == 0.0:
        return 0.0
    if eps < 1.0e-3:
        squared = eps * eps
        return float(squared / 8.0 - squared * squared / 576.0)
    if eps > 50.0:
        return float(eps - math.log(eps) - 1.0 + math.log1p(-math.exp(-eps)))
    expm1 = math.expm1(eps)
    ratio_minus_one = expm1 / eps - 1.0
    if abs(ratio_minus_one) < 0.125:
        # log(1 + u) - u / (1 + u), expanded after its linear terms cancel.
        # This avoids losing low-budget rho to subtraction of values near one.
        power = ratio_minus_one * ratio_minus_one
        value = 0.0
        for order in range(2, 40):
            coefficient = ((-1.0) ** order) * (order - 1.0) / order
            value += coefficient * power
            power *= ratio_minus_one
    else:
        value = math.log1p(ratio_minus_one) - ratio_minus_one / (
            1.0 + ratio_minus_one
        )
    if value < 0.0 and value > -1.0e-14:
        value = 0.0
    if not math.isfinite(value) or value < 0.0:
        raise ArithmeticError("bounded-range zCDP conversion became invalid")
    return float(value)


def bounded_range_epsilon(rho: float) -> float:
    """Invert :func:`bounded_range_rho` by monotone bisection."""
    target = float(rho)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("rho must be finite and non-negative")
    if target == 0.0:
        return 0.0
    lower = 0.0
    upper = max(1.0, math.sqrt(8.0 * target) * 1.05)
    while bounded_range_rho(upper) < target:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ArithmeticError("could not bracket bounded-range epsilon")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if bounded_range_rho(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    return float(0.5 * (lower + upper))


@dataclass(frozen=True)
class ZCDPLedgerEntry:
    label: str
    mechanism: str
    rho: float
    public_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mechanism": self.mechanism,
            "rho": self.rho,
            "public_metadata": dict(self.public_metadata),
        }


class ZCDPPrivacyFilter:
    """Fail-closed zCDP ledger whose actual spend is the source of truth."""

    def __init__(self, rho_limit: float):
        limit = float(rho_limit)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("rho_limit must be finite and positive")
        self._rho_limit = limit
        self._entries: list[ZCDPLedgerEntry] = []

    @property
    def rho_limit(self) -> float:
        return self._rho_limit

    @property
    def rho_spent(self) -> float:
        return float(math.fsum(entry.rho for entry in self._entries))

    @property
    def rho_remaining(self) -> float:
        return float(max(0.0, self.rho_limit - self.rho_spent))

    @property
    def entries(self) -> tuple[ZCDPLedgerEntry, ...]:
        return tuple(self._entries)

    def spend(
        self,
        *,
        label: str,
        mechanism: str,
        rho: float,
        public_metadata: dict[str, Any] | None = None,
    ) -> None:
        amount = float(rho)
        if not label or not mechanism:
            raise ValueError("ledger label and mechanism must be non-empty")
        if not math.isfinite(amount) or amount <= 0.0:
            raise ValueError("ledger rho must be finite and positive")
        proposed = self.rho_spent + amount
        tolerance = 1.0e-12 * max(1.0, self.rho_limit)
        if proposed > self.rho_limit + tolerance:
            raise RuntimeError(
                f"privacy filter would overspend rho: proposed={proposed}, "
                f"limit={self.rho_limit}"
            )
        self._entries.append(
            ZCDPLedgerEntry(
                label=str(label),
                mechanism=str(mechanism),
                rho=amount,
                public_metadata=dict(public_metadata or {}),
            )
        )

    def to_public_dict(self, *, delta: float) -> dict[str, Any]:
        spent = self.rho_spent
        return {
            "accounting": "zcdp_actual_spend_v1",
            "rho_limit": self.rho_limit,
            "rho_spent": spent,
            "rho_remaining": self.rho_remaining,
            "delta": float(delta),
            "epsilon_from_actual_spend": zcdp_epsilon(spent, float(delta)),
            "entries": [entry.to_dict() for entry in self._entries],
        }
