from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet
from qdte.rce.relaxed import RestrictedMixtureRCEResult


CDWF_DUAL_METHOD = "certified_restricted_minimum_norm_rce_dual_v1"


@dataclass(frozen=True)
class CertifiedRestrictedRCEDual:
    ellipsoid_weight: float
    tube_positive: np.ndarray
    tube_negative: np.ndarray
    face_slack: float
    certificate: dict[str, Any]
    eligible: bool
    failure_reasons: tuple[str, ...]
    method: str = CDWF_DUAL_METHOD

    def __post_init__(self) -> None:
        positive = np.asarray(self.tube_positive, dtype=np.float64)
        negative = np.asarray(self.tube_negative, dtype=np.float64)
        if self.method != CDWF_DUAL_METHOD:
            raise ValueError(f"Unsupported C3 restricted dual method {self.method!r}")
        if positive.ndim != 1 or negative.shape != positive.shape:
            raise ValueError("C3 tube multipliers must be matching vectors")
        if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
            raise ValueError("C3 tube multipliers must be finite")
        if np.any(positive < 0.0) or np.any(negative < 0.0):
            raise ValueError("C3 tube multipliers must be nonnegative")
        if not math.isfinite(float(self.ellipsoid_weight)) or self.ellipsoid_weight < 0.0:
            raise ValueError("C3 ellipsoid multiplier must be finite and nonnegative")
        if not math.isfinite(float(self.face_slack)) or self.face_slack < 0.0:
            raise ValueError("C3 face slack must be finite and nonnegative")
        positive = positive.copy()
        negative = negative.copy()
        positive.setflags(write=False)
        negative.setflags(write=False)
        object.__setattr__(self, "tube_positive", positive)
        object.__setattr__(self, "tube_negative", negative)
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))

    @property
    def dimension(self) -> int:
        return int(len(self.tube_positive))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "eligible": self.eligible,
            "failure_reasons": list(self.failure_reasons),
            "ellipsoid_weight": self.ellipsoid_weight,
            "tube_positive": self.tube_positive.tolist(),
            "tube_negative": self.tube_negative.tolist(),
            "face_slack": self.face_slack,
            "certificate": dict(self.certificate),
        }


def extract_certified_restricted_dual(
    result: RestrictedMixtureRCEResult,
    confidence: RCEConfidenceSet,
) -> CertifiedRestrictedRCEDual:
    certificate = dict(result.restricted_dual_certificate)
    ordering = certificate.get("constraint_order")
    minimum_norm = certificate.get("minimum_norm_dual")
    if not isinstance(ordering, dict) or not isinstance(minimum_norm, dict):
        raise ValueError("Restricted RCE result does not contain a C3 dual certificate")
    multipliers = np.asarray(minimum_norm.get("multipliers", []), dtype=np.float64)
    tube_indices = np.asarray(ordering.get("tube_coordinate_indices", []), dtype=np.int64)
    expected = 1 + 2 * len(tube_indices)
    if multipliers.shape != (expected,):
        raise ValueError("Restricted RCE multiplier vector has an invalid dimension")
    if np.any(tube_indices < 0) or np.any(tube_indices >= confidence.dimension):
        raise ValueError("Restricted RCE tube ordering contains an invalid coordinate")
    positive_start = int(ordering.get("tube_positive_start", -1))
    negative_start = int(ordering.get("tube_negative_start", -1))
    if positive_start != 1 or negative_start != 1 + len(tube_indices):
        raise ValueError("Restricted RCE constraint ordering is not canonical")
    tube_positive = np.zeros(confidence.dimension, dtype=np.float64)
    tube_negative = np.zeros(confidence.dimension, dtype=np.float64)
    tube_positive[tube_indices] = multipliers[
        positive_start : positive_start + len(tube_indices)
    ]
    tube_negative[tube_indices] = multipliers[
        negative_start : negative_start + len(tube_indices)
    ]
    # The restricted solver records h_2/c_2 <= 0, whereas C3's pressure
    # definition uses the unnormalized ellipsoid h_2 <= 0.
    ellipsoid_weight = float(multipliers[0]) / confidence.squared_discrepancy_threshold
    residual = np.asarray(result.residual, dtype=np.float64)
    face_slack = float(ordering.get("face_slack", math.inf))
    normalized_ellipsoid = (
        confidence.squared_discrepancy(residual)
        / confidence.squared_discrepancy_threshold
        - 1.0
        - face_slack
    )
    bounds = confidence.coordinate_bounds * (1.0 + face_slack)
    stochastic = confidence.tube_mask
    tube_violation = max(
        0.0,
        float(np.max(residual[stochastic] - bounds[stochastic], initial=-math.inf)),
        float(np.max(-residual[stochastic] - bounds[stochastic], initial=-math.inf)),
    )
    checks = {
        "simplex_violation": abs(float(np.sum(result.component_weights)) - 1.0),
        "global_ellipsoid_violation": max(0.0, normalized_ellipsoid),
        "tube_violation": tube_violation,
        "dual_negativity": float(minimum_norm.get("dual_negativity", math.inf)),
        "stationarity_infinity_norm": float(
            minimum_norm.get("stationarity_residual", math.inf)
        ),
        "complementarity_residual": float(
            minimum_norm.get("complementarity_residual", math.inf)
        ),
        "relative_primal_dual_gap": float(
            minimum_norm.get("relative_primal_dual_gap", math.inf)
        ),
        "face_slack": face_slack,
    }
    reasons: list[str] = []
    if minimum_norm.get("success") is not True:
        reasons.append("minimum_norm_dual_failed")
    tolerances = {
        "simplex_violation": 1.0e-10,
        "global_ellipsoid_violation": 1.0e-8,
        "tube_violation": 1.0e-8,
        "dual_negativity": 1.0e-10,
        "stationarity_infinity_norm": 1.0e-7,
        "complementarity_residual": 1.0e-7,
        "relative_primal_dual_gap": 1.0e-6,
        "face_slack": 1.0e-8,
    }
    for name, tolerance in tolerances.items():
        if not math.isfinite(checks[name]) or checks[name] > tolerance:
            reasons.append(f"{name}_exceeds_{tolerance:g}")
    public_certificate = {
        **checks,
        "tolerances": tolerances,
        "restricted_support_only": True,
        "global_pricing_certificate": False,
        "minimum_norm_solver": dict(minimum_norm),
    }
    return CertifiedRestrictedRCEDual(
        ellipsoid_weight=ellipsoid_weight,
        tube_positive=tube_positive,
        tube_negative=tube_negative,
        face_slack=face_slack,
        certificate=public_certificate,
        eligible=not reasons,
        failure_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class CDWFPressureResult:
    pressure_by_block: dict[str, float]
    formula_by_block: dict[str, dict[str, float]]
    autodiff_by_block: dict[str, float]
    maximum_autodiff_error: float
    autodiff_audit_passed: bool

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "pressure_by_block": dict(self.pressure_by_block),
            "formula_by_block": {
                name: dict(values) for name, values in self.formula_by_block.items()
            },
            "autodiff_by_block": dict(self.autodiff_by_block),
            "maximum_autodiff_error": self.maximum_autodiff_error,
            "autodiff_audit_passed": self.autodiff_audit_passed,
        }


def compute_cdwf_block_pressures(
    *,
    residual: np.ndarray,
    confidence: RCEConfidenceSet,
    dual: CertifiedRestrictedRCEDual,
    block_slices: Mapping[str, slice],
    rho_by_block: Mapping[str, float],
) -> CDWFPressureResult:
    vector = np.asarray(residual, dtype=np.float64)
    if vector.shape != (confidence.dimension,) or dual.dimension != confidence.dimension:
        raise ValueError("C3 pressure inputs must share the confidence dimension")
    if confidence.precision_diagonal is None:
        raise ValueError("C3-CDWF-v1 requires diagonal combined coefficient precision")
    if set(block_slices) != set(rho_by_block):
        raise ValueError("C3 block slices and rho maps must have identical support")
    names = tuple(block_slices)
    coverage = np.zeros(confidence.dimension, dtype=np.int32)
    for name in names:
        local = np.arange(confidence.dimension)[block_slices[name]]
        if len(local) == 0:
            raise ValueError(f"C3 block {name!r} has an empty coefficient slice")
        coverage[local] += 1
    if not np.all(coverage == 1):
        raise ValueError("C3 block slices must partition the coefficient space")
    precision = np.asarray(confidence.precision_diagonal, dtype=np.float64)
    widths = confidence.coordinate_bounds * (1.0 + dual.face_slack)
    formula: dict[str, dict[str, float]] = {}
    pressure: dict[str, float] = {}
    for name in names:
        local = block_slices[name]
        ellipsoid = dual.ellipsoid_weight * float(
            np.sum(vector[local] ** 2 * precision[local], dtype=np.float64)
        )
        tube = 0.5 * float(
            np.sum(
                (dual.tube_positive[local] + dual.tube_negative[local])
                * widths[local],
                dtype=np.float64,
            )
        )
        value = max(0.0, ellipsoid + tube)
        formula[name] = {
            "ellipsoid": ellipsoid,
            "tube": tube,
            "total": value,
        }
        pressure[name] = value

    rho = np.asarray([float(rho_by_block[name]) for name in names], dtype=np.float64)
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("C3 block rho values must be finite and positive")
    variance_constants: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    positive_blocks: list[np.ndarray] = []
    negative_blocks: list[np.ndarray] = []
    for index, name in enumerate(names):
        local = block_slices[name]
        variance_constants.append(
            confidence.marginal_variances[local] * rho[index]
        )
        residual_blocks.append(vector[local])
        positive_blocks.append(dual.tube_positive[local])
        negative_blocks.append(dual.tube_negative[local])

    def lagrangian_constraints(log_rho: jax.Array) -> jax.Array:
        ellipsoid_value = jnp.asarray(
            -confidence.squared_discrepancy_threshold * (1.0 + dual.face_slack),
            dtype=jnp.float64,
        )
        tube_value = jnp.asarray(0.0, dtype=jnp.float64)
        for index in range(len(names)):
            local_rho = jnp.exp(log_rho[index])
            constants = jnp.asarray(variance_constants[index], dtype=jnp.float64)
            local_residual = jnp.asarray(residual_blocks[index], dtype=jnp.float64)
            local_width = (
                confidence.coordinate_standardized_threshold
                * jnp.sqrt(constants / local_rho)
                * (1.0 + dual.face_slack)
            )
            ellipsoid_value = ellipsoid_value + jnp.sum(
                local_residual * local_residual * local_rho / constants
            )
            tube_value = tube_value + jnp.sum(
                jnp.asarray(positive_blocks[index], dtype=jnp.float64)
                * (local_residual - local_width)
                + jnp.asarray(negative_blocks[index], dtype=jnp.float64)
                * (-local_residual - local_width)
            )
        return dual.ellipsoid_weight * ellipsoid_value + tube_value

    with jax.enable_x64():
        autodiff_values = np.asarray(
            jax.grad(lagrangian_constraints)(jnp.log(jnp.asarray(rho))),
            dtype=np.float64,
        )
    autodiff = {
        name: max(0.0, float(value))
        for name, value in zip(names, autodiff_values, strict=True)
    }
    errors = np.asarray(
        [abs(autodiff[name] - pressure[name]) for name in names],
        dtype=np.float64,
    )
    scale = max(1.0, max(pressure.values(), default=0.0))
    maximum_error = float(np.max(errors, initial=0.0))
    return CDWFPressureResult(
        pressure_by_block=pressure,
        formula_by_block=formula,
        autodiff_by_block=autodiff,
        maximum_autodiff_error=maximum_error,
        autodiff_audit_passed=maximum_error <= 1.0e-9 * scale,
    )


__all__ = [
    "CDWF_DUAL_METHOD",
    "CDWFPressureResult",
    "CertifiedRestrictedRCEDual",
    "compute_cdwf_block_pressures",
    "extract_certified_restricted_dual",
]
