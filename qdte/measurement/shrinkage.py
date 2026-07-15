from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdte.measurement.factorization import HierarchicalInteractionTranscript


SHRINKAGE_METHOD = "positive_part_block_james_stein_v1"
MIN_PAIR_DIMENSION = 3
ANALYTIC_MIN_RELATIVE_KINK_MARGIN = 0.05


@dataclass(frozen=True)
class InteractionShrinkageBlock:
    name: str
    kind: str
    coefficient_shape: tuple[int, ...]
    variance: float
    raw_coefficients: np.ndarray
    shrunk_coefficients: np.ndarray
    factor: float
    threshold_numerator: float
    norm_squared: float
    regime: str
    relative_kink_margin: float
    analytic_stable: bool

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_coefficients, dtype=np.float64)
        shrunk = np.asarray(self.shrunk_coefficients, dtype=np.float64)
        if raw.shape != self.coefficient_shape or shrunk.shape != self.coefficient_shape:
            raise ValueError(f"Shrinkage block {self.name!r} has inconsistent coefficient shapes")
        if raw.size == 0 or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(shrunk)):
            raise ValueError(f"Shrinkage block {self.name!r} coefficients must be finite and non-empty")
        if not np.isfinite(self.variance) or float(self.variance) <= 0.0:
            raise ValueError(f"Shrinkage block {self.name!r} variance must be positive and finite")
        if not np.isfinite(self.factor) or not 0.0 <= float(self.factor) <= 1.0:
            raise ValueError(f"Shrinkage block {self.name!r} factor must lie in [0, 1]")
        if not np.isfinite(self.threshold_numerator) or float(self.threshold_numerator) < 0.0:
            raise ValueError(f"Shrinkage block {self.name!r} threshold must be finite and nonnegative")
        if not np.isfinite(self.norm_squared) or float(self.norm_squared) < 0.0:
            raise ValueError(f"Shrinkage block {self.name!r} norm must be finite and nonnegative")
        if not np.isfinite(self.relative_kink_margin) or not 0.0 <= float(
            self.relative_kink_margin
        ) <= 1.0:
            raise ValueError(f"Shrinkage block {self.name!r} kink margin must lie in [0, 1]")
        raw = raw.copy()
        shrunk = shrunk.copy()
        raw.setflags(write=False)
        shrunk.setflags(write=False)
        object.__setattr__(self, "raw_coefficients", raw)
        object.__setattr__(self, "shrunk_coefficients", shrunk)

    @property
    def dimension(self) -> int:
        return int(np.prod(self.coefficient_shape, dtype=np.int64))

    @property
    def raw_vector(self) -> np.ndarray:
        return self.raw_coefficients.reshape(-1)

    def _vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite with shape ({self.dimension},)")
        return vector

    def jacobian_matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        if self.regime in {"identity_oneway", "identity_small_pair"}:
            return vector.copy()
        if self.regime in {"positive_part_zero", "positive_part_kink"}:
            return np.zeros_like(vector)
        if self.regime != "positive_part_active":
            raise RuntimeError(f"Unknown shrinkage regime {self.regime!r}")
        raw = self.raw_vector
        rank_one_scale = 2.0 * self.threshold_numerator / (self.norm_squared**2)
        return self.factor * vector + rank_one_scale * raw * float(raw @ vector)

    def jacobian_matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension or not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(f"values must be finite with shape (n, {self.dimension})")
        if self.regime in {"identity_oneway", "identity_small_pair"}:
            return matrix.copy()
        if self.regime in {"positive_part_zero", "positive_part_kink"}:
            return np.zeros_like(matrix)
        raw = self.raw_vector
        rank_one_scale = 2.0 * self.threshold_numerator / (self.norm_squared**2)
        return self.factor * matrix + rank_one_scale * np.outer(matrix @ raw, raw)

    def covariance_matvec(self, values: np.ndarray) -> np.ndarray:
        first = self.jacobian_matvec(values)
        return float(self.variance) * self.jacobian_matvec(first)

    def precision_parameters(self) -> tuple[float, float, np.ndarray]:
        """Return isotropic, rank-one correction, and unit direction for Cov(g(z))^+."""
        if self.regime in {"positive_part_zero", "positive_part_kink"}:
            return 0.0, 0.0, np.zeros(self.dimension, dtype=np.float64)
        if self.regime in {"identity_oneway", "identity_small_pair"}:
            return 1.0 / float(self.variance), 0.0, np.zeros(
                self.dimension, dtype=np.float64
            )
        raw = self.raw_vector
        unit = raw / np.sqrt(self.norm_squared)
        perpendicular = 1.0 / (float(self.variance) * self.factor**2)
        parallel_jacobian = 2.0 - self.factor
        parallel = 1.0 / (float(self.variance) * parallel_jacobian**2)
        return perpendicular, parallel - perpendicular, unit

    def precision_matvec(self, values: np.ndarray) -> np.ndarray:
        vector = self._vector(values, name="values")
        isotropic, directional, unit = self.precision_parameters()
        return isotropic * vector + directional * unit * float(unit @ vector)

    def dense_jacobian(self) -> np.ndarray:
        return self.jacobian_matvec_many(np.eye(self.dimension, dtype=np.float64)).T

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "coefficient_shape": list(self.coefficient_shape),
            "dimension": self.dimension,
            "variance": float(self.variance),
            "norm_squared": float(self.norm_squared),
            "threshold_numerator": float(self.threshold_numerator),
            "factor": float(self.factor),
            "regime": self.regime,
            "relative_kink_margin": float(self.relative_kink_margin),
            "analytic_stable": bool(self.analytic_stable),
        }


@dataclass(frozen=True)
class InteractionShrinkageResult:
    blocks: tuple[InteractionShrinkageBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("Interaction shrinkage requires at least one coefficient block")
        names = [block.name for block in self.blocks]
        if len(names) != len(set(names)):
            raise ValueError("Interaction shrinkage block names must be unique")

    @property
    def coefficient_dimension(self) -> int:
        return int(sum(block.dimension for block in self.blocks))

    @property
    def analytic_stable(self) -> bool:
        return all(block.analytic_stable for block in self.blocks)

    @property
    def effective_rank(self) -> int:
        return int(
            sum(
                block.dimension
                for block in self.blocks
                if block.regime not in {"positive_part_zero", "positive_part_kink"}
            )
        )

    def block(self, name: str) -> InteractionShrinkageBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)

    def _vector(self, values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.coefficient_dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} must be finite with shape ({self.coefficient_dimension},)"
            )
        return vector

    def _apply_blocks(self, values: np.ndarray, operation: str) -> np.ndarray:
        vector = self._vector(values, name="values")
        output = np.empty_like(vector)
        offset = 0
        for block in self.blocks:
            end = offset + block.dimension
            method = getattr(block, operation)
            output[offset:end] = method(vector[offset:end])
            offset = end
        return output

    def jacobian_matvec(self, values: np.ndarray) -> np.ndarray:
        return self._apply_blocks(values, "jacobian_matvec")

    def covariance_matvec(self, values: np.ndarray) -> np.ndarray:
        return self._apply_blocks(values, "covariance_matvec")

    def precision_matvec(self, values: np.ndarray) -> np.ndarray:
        return self._apply_blocks(values, "precision_matvec")

    def precision_matvec_many(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.coefficient_dimension or not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                f"values must be finite with shape (n, {self.coefficient_dimension})"
            )
        return np.stack([self.precision_matvec(row) for row in matrix], axis=0)

    def shrunk_transcript(
        self,
        transcript: HierarchicalInteractionTranscript,
    ) -> HierarchicalInteractionTranscript:
        expected = tuple(block.name for block in transcript.strategy.blocks)
        actual = tuple(block.name for block in self.blocks)
        if actual != expected:
            raise ValueError("Shrinkage block order does not match the interaction transcript")
        components: dict[str, np.ndarray] = {}
        for strategy_block, result_block in zip(
            transcript.strategy.blocks,
            self.blocks,
            strict=True,
        ):
            if result_block.coefficient_shape != strategy_block.coefficient_shape:
                raise ValueError(f"Shrinkage shape mismatch for {strategy_block.name!r}")
            components[strategy_block.name] = result_block.shrunk_coefficients.copy()
        return HierarchicalInteractionTranscript(
            strategy=transcript.strategy,
            public_total=int(transcript.public_total),
            noisy_components=components,
            component_variances=dict(transcript.component_variances),
            rho_by_block=dict(transcript.rho_by_block),
            rho_total=float(transcript.rho_total),
            rho_spent=float(transcript.rho_spent),
            allocation_mode=transcript.allocation_mode,
            adjacency=transcript.adjacency,
        )

    def diagnostics(self) -> dict[str, Any]:
        pair_blocks = [block for block in self.blocks if block.kind == "pair_interaction"]
        affected = [block for block in pair_blocks if block.dimension >= MIN_PAIR_DIMENSION]
        factors = np.asarray([block.factor for block in affected], dtype=np.float64)
        correction_squared = float(
            sum(
                np.sum(
                    (block.shrunk_coefficients - block.raw_coefficients)
                    * (block.shrunk_coefficients - block.raw_coefficients)
                )
                for block in self.blocks
            )
        )
        return {
            "enabled": True,
            "method": SHRINKAGE_METHOD,
            "oneway_rule": "identity",
            "small_pair_rule": "identity",
            "minimum_pair_dimension": MIN_PAIR_DIMENSION,
            "analytic_min_relative_kink_margin": ANALYTIC_MIN_RELATIVE_KINK_MARGIN,
            "num_blocks": len(self.blocks),
            "num_pair_blocks": len(pair_blocks),
            "num_affected_pair_blocks": len(affected),
            "num_active_shrinkage_blocks": sum(
                block.regime == "positive_part_active" for block in affected
            ),
            "num_zero_shrinkage_blocks": sum(
                block.regime in {"positive_part_zero", "positive_part_kink"}
                for block in affected
            ),
            "affected_coefficient_count": int(sum(block.dimension for block in affected)),
            "coefficient_dimension": self.coefficient_dimension,
            "effective_rank": self.effective_rank,
            "factor_min": float(np.min(factors)) if factors.size else 1.0,
            "factor_mean": float(np.mean(factors)) if factors.size else 1.0,
            "factor_max": float(np.max(factors)) if factors.size else 1.0,
            "coefficient_correction_l2": float(np.sqrt(correction_squared)),
            "no_op": not affected,
            "analytic_covariance_stable": self.analytic_stable,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": self.diagnostics(),
            "blocks": [block.to_public_dict() for block in self.blocks],
        }


def positive_part_block_james_stein(
    transcript: HierarchicalInteractionTranscript,
) -> InteractionShrinkageResult:
    """Shrink pair-interaction blocks using only the released transcript."""
    if transcript.adjacency != "add_remove":
        raise ValueError("Interaction shrinkage currently requires add_remove adjacency")
    blocks: list[InteractionShrinkageBlock] = []
    for strategy_block in transcript.strategy.blocks:
        raw = np.asarray(
            transcript.noisy_components[strategy_block.name],
            dtype=np.float64,
        )
        variance = float(transcript.component_variances[strategy_block.name])
        dimension = strategy_block.dimension
        norm_squared = float(np.sum(raw * raw))
        threshold = 0.0
        factor = 1.0
        relative_margin = 1.0
        analytic_stable = True
        if strategy_block.kind == "oneway_contrast":
            regime = "identity_oneway"
        elif strategy_block.kind != "pair_interaction":
            raise ValueError(f"Unsupported shrinkage block kind {strategy_block.kind!r}")
        elif dimension < MIN_PAIR_DIMENSION:
            regime = "identity_small_pair"
        else:
            threshold = float(dimension - 2) * variance
            denominator = max(norm_squared, threshold, np.finfo(np.float64).tiny)
            relative_margin = abs(norm_squared - threshold) / denominator
            if np.isclose(norm_squared, threshold, rtol=0.0, atol=np.finfo(np.float64).eps * denominator):
                factor = 0.0
                regime = "positive_part_kink"
                analytic_stable = False
            elif norm_squared <= threshold:
                factor = 0.0
                regime = "positive_part_zero"
                analytic_stable = False
            else:
                factor = max(0.0, 1.0 - threshold / norm_squared)
                regime = "positive_part_active"
                analytic_stable = bool(
                    relative_margin >= ANALYTIC_MIN_RELATIVE_KINK_MARGIN
                )
        blocks.append(
            InteractionShrinkageBlock(
                name=strategy_block.name,
                kind=strategy_block.kind,
                coefficient_shape=strategy_block.coefficient_shape,
                variance=variance,
                raw_coefficients=raw,
                shrunk_coefficients=factor * raw,
                factor=float(factor),
                threshold_numerator=float(threshold),
                norm_squared=norm_squared,
                regime=regime,
                relative_kink_margin=float(relative_margin),
                analytic_stable=analytic_stable,
            )
        )
    return InteractionShrinkageResult(blocks=tuple(blocks))


def validate_frozen_shrinkage_artifact(
    result: InteractionShrinkageResult,
    payload: dict[str, Any],
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Interaction shrinkage artifact must be a mapping")
    diagnostics = payload.get("diagnostics")
    raw_blocks = payload.get("blocks")
    if not isinstance(diagnostics, dict) or not isinstance(raw_blocks, list):
        raise ValueError("Interaction shrinkage artifact is missing diagnostics or blocks")
    expected_diagnostics = result.diagnostics()
    frozen_fields = (
        "method",
        "oneway_rule",
        "small_pair_rule",
        "minimum_pair_dimension",
        "analytic_min_relative_kink_margin",
        "num_blocks",
        "coefficient_dimension",
    )
    for field in frozen_fields:
        if diagnostics.get(field) != expected_diagnostics[field]:
            raise ValueError(f"Interaction shrinkage artifact mismatch for {field!r}")
    if len(raw_blocks) != len(result.blocks):
        raise ValueError("Interaction shrinkage artifact block count mismatch")
    for expected, raw in zip(result.blocks, raw_blocks, strict=True):
        if not isinstance(raw, dict):
            raise ValueError("Interaction shrinkage artifact block must be a mapping")
        if raw.get("name") != expected.name or raw.get("regime") != expected.regime:
            raise ValueError(f"Interaction shrinkage artifact block mismatch for {expected.name!r}")
        if int(raw.get("dimension", -1)) != expected.dimension:
            raise ValueError(f"Interaction shrinkage artifact dimension mismatch for {expected.name!r}")
        factor = float(raw.get("factor", float("nan")))
        if not np.isclose(factor, expected.factor, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(f"Interaction shrinkage artifact factor mismatch for {expected.name!r}")
