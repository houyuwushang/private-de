from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from qdte.rce.confidence_set import RCEConfidenceSet


STREAMWISE_RCE_METHOD = "adaptive_safe_streamwise_rce_confidence_v1"


def _readonly(values: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RCECoefficientStream:
    name: str
    coefficient_indices: np.ndarray
    released_target: np.ndarray
    variances: np.ndarray
    alpha_l2: float
    alpha_linf: float
    stage: str
    round_index: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RCE stream name must be non-empty")
        indices = np.asarray(self.coefficient_indices, dtype=np.int64)
        target = np.asarray(self.released_target, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("RCE stream indices must be a non-empty vector")
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
            raise ValueError("RCE stream indices must be distinct and nonnegative")
        if target.shape != indices.shape or variances.shape != indices.shape:
            raise ValueError("RCE stream target and variances must match its indices")
        if not np.all(np.isfinite(target)):
            raise ValueError("RCE stream target must be finite")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("RCE stream variances must be finite and positive")
        if self.stage not in {"base", "refinement"}:
            raise ValueError("RCE stream stage must be base or refinement")
        if self.stage == "base" and int(self.round_index) != -1:
            raise ValueError("The base RCE stream must use round_index=-1")
        if self.stage == "refinement" and int(self.round_index) < 0:
            raise ValueError("Refinement RCE streams require a nonnegative round index")
        confidence = RCEConfidenceSet.from_diagonal_variances(
            variances,
            alpha_l2=float(self.alpha_l2),
            alpha_linf=float(self.alpha_linf),
        )
        object.__setattr__(self, "coefficient_indices", _readonly(indices, dtype=np.int64))
        object.__setattr__(self, "released_target", _readonly(target, dtype=np.float64))
        object.__setattr__(self, "variances", _readonly(variances, dtype=np.float64))
        object.__setattr__(self, "_confidence", confidence)

    @property
    def confidence(self) -> RCEConfidenceSet:
        return self._confidence

    @property
    def dimension(self) -> int:
        return int(len(self.coefficient_indices))

    def residual(self, canonical_answer: np.ndarray) -> np.ndarray:
        answer = np.asarray(canonical_answer, dtype=np.float64)
        if answer.ndim != 1 or np.max(self.coefficient_indices) >= len(answer):
            raise ValueError("Canonical answer does not cover the RCE stream indices")
        if not np.all(np.isfinite(answer)):
            raise ValueError("Canonical answer must be finite")
        return self.released_target - answer[self.coefficient_indices]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "round_index": self.round_index,
            "coefficient_indices": self.coefficient_indices.tolist(),
            "released_target": self.released_target.tolist(),
            "variances": self.variances.tolist(),
            "alpha_l2": self.alpha_l2,
            "alpha_linf": self.alpha_linf,
            "confidence": self.confidence.diagnostics(),
        }

    @classmethod
    def from_public_dict(cls, data: dict[str, Any]) -> RCECoefficientStream:
        return cls(
            name=str(data["name"]),
            stage=str(data["stage"]),
            round_index=int(data["round_index"]),
            coefficient_indices=np.asarray(data["coefficient_indices"], dtype=np.int64),
            released_target=np.asarray(data["released_target"], dtype=np.float64),
            variances=np.asarray(data["variances"], dtype=np.float64),
            alpha_l2=float(data["alpha_l2"]),
            alpha_linf=float(data["alpha_linf"]),
        )


@dataclass(frozen=True)
class StreamwiseRCEConfidenceSet:
    canonical_dimension: int
    combined_target: np.ndarray
    streams: tuple[RCECoefficientStream, ...]
    alpha_total: float = 0.05
    method: str = STREAMWISE_RCE_METHOD

    def __post_init__(self) -> None:
        dimension = int(self.canonical_dimension)
        target = np.asarray(self.combined_target, dtype=np.float64)
        streams = tuple(self.streams)
        if self.method != STREAMWISE_RCE_METHOD:
            raise ValueError(f"Unsupported streamwise RCE method {self.method!r}")
        if dimension <= 0 or target.shape != (dimension,) or not np.all(np.isfinite(target)):
            raise ValueError("Streamwise RCE combined target must match a positive dimension")
        if not streams or streams[0].stage != "base":
            raise ValueError("Streamwise RCE requires the base stream first")
        if len({stream.name for stream in streams}) != len(streams):
            raise ValueError("Streamwise RCE stream names must be unique")
        base_indices = streams[0].coefficient_indices
        if not np.array_equal(base_indices, np.arange(dimension, dtype=np.int64)):
            raise ValueError("The base RCE stream must cover the full canonical coefficient space")
        for stream in streams:
            if np.max(stream.coefficient_indices) >= dimension:
                raise ValueError("RCE stream index exceeds the canonical dimension")
        refinement_rounds = [
            stream.round_index for stream in streams if stream.stage == "refinement"
        ]
        if refinement_rounds != list(range(len(refinement_rounds))):
            raise ValueError("Refinement RCE streams must use contiguous ordered rounds")
        spent_alpha = math.fsum(
            stream.alpha_l2 + stream.alpha_linf for stream in streams
        )
        tolerance = 1.0e-12 * max(1.0, float(self.alpha_total))
        if spent_alpha > float(self.alpha_total) + tolerance:
            raise ValueError("Streamwise RCE alpha spending exceeds its declared total")
        payload = [stream.to_public_dict() for stream in streams]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "canonical_dimension", dimension)
        object.__setattr__(self, "combined_target", _readonly(target, dtype=np.float64))
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "_stream_hash", digest)

    @property
    def stream_hash(self) -> str:
        return self._stream_hash

    @property
    def confidence_level(self) -> float:
        return 1.0 - math.fsum(
            stream.alpha_l2 + stream.alpha_linf for stream in self.streams
        )

    def canonical_answer_from_combined_residual(
        self,
        combined_residual: np.ndarray,
    ) -> np.ndarray:
        residual = np.asarray(combined_residual, dtype=np.float64)
        if residual.shape != (self.canonical_dimension,) or not np.all(np.isfinite(residual)):
            raise ValueError("Combined residual must match the canonical dimension")
        return self.combined_target - residual

    def residuals(self, canonical_answer: np.ndarray) -> tuple[np.ndarray, ...]:
        answer = np.asarray(canonical_answer, dtype=np.float64)
        if answer.shape != (self.canonical_dimension,) or not np.all(np.isfinite(answer)):
            raise ValueError("Canonical answer must match the streamwise RCE dimension")
        return tuple(stream.residual(answer) for stream in self.streams)

    def evaluate_answer(
        self,
        canonical_answer: np.ndarray,
        *,
        tolerance: float = 1.0e-12,
    ) -> dict[str, Any]:
        evaluations = [
            stream.confidence.evaluate(residual, tolerance=float(tolerance))
            for stream, residual in zip(
                self.streams,
                self.residuals(canonical_answer),
                strict=True,
            )
        ]
        return {
            "inside": all(evaluation.inside for evaluation in evaluations),
            "maximum_slack": max(evaluation.slack for evaluation in evaluations),
            "maximum_ellipsoid_ratio": max(
                evaluation.ellipsoid_ratio for evaluation in evaluations
            ),
            "maximum_tube_ratio": max(evaluation.tube_ratio for evaluation in evaluations),
            "num_streams": len(self.streams),
            "streams": {
                stream.name: evaluation.to_dict()
                for stream, evaluation in zip(self.streams, evaluations, strict=True)
            },
            # Compatibility aliases used by the existing RCE engine log and
            # integer-incumbent interface.
            "slack": max(evaluation.slack for evaluation in evaluations),
            "ellipsoid_ratio": max(
                evaluation.ellipsoid_ratio for evaluation in evaluations
            ),
            "tube_ratio": max(evaluation.tube_ratio for evaluation in evaluations),
        }

    def evaluate_combined_residual(
        self,
        combined_residual: np.ndarray,
        *,
        tolerance: float = 1.0e-12,
    ) -> dict[str, Any]:
        return self.evaluate_answer(
            self.canonical_answer_from_combined_residual(combined_residual),
            tolerance=float(tolerance),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "canonical_dimension": self.canonical_dimension,
            "combined_target": self.combined_target.tolist(),
            "alpha_total": self.alpha_total,
            "alpha_spent": math.fsum(
                stream.alpha_l2 + stream.alpha_linf for stream in self.streams
            ),
            "confidence_level_union_bound": self.confidence_level,
            "stream_hash": self.stream_hash,
            "streams": [stream.to_public_dict() for stream in self.streams],
        }

    @classmethod
    def from_public_dict(cls, data: dict[str, Any]) -> StreamwiseRCEConfidenceSet:
        result = cls(
            method=str(data["method"]),
            canonical_dimension=int(data["canonical_dimension"]),
            combined_target=np.asarray(data["combined_target"], dtype=np.float64),
            alpha_total=float(data["alpha_total"]),
            streams=tuple(
                RCECoefficientStream.from_public_dict(stream)
                for stream in data["streams"]
            ),
        )
        if result.stream_hash != str(data.get("stream_hash", result.stream_hash)):
            raise ValueError("Serialized streamwise RCE hash does not match its content")
        return result


def build_cdwf_streamwise_confidence(
    *,
    combined_target: np.ndarray,
    base_target: np.ndarray,
    base_variances: np.ndarray,
    refinement_targets: Sequence[np.ndarray],
    refinement_variances: Sequence[np.ndarray],
    refinement_indices: Sequence[np.ndarray],
    alpha_total: float = 0.05,
) -> StreamwiseRCEConfidenceSet:
    target = np.asarray(combined_target, dtype=np.float64)
    if len(refinement_targets) != len(refinement_variances) or len(
        refinement_targets
    ) != len(refinement_indices):
        raise ValueError("CDWF refinement stream inputs must have matching lengths")
    rounds = len(refinement_targets)
    if rounds <= 0:
        raise ValueError("CDWF streamwise confidence requires at least one refinement round")
    streams = [
        RCECoefficientStream(
            name="base",
            coefficient_indices=np.arange(len(target), dtype=np.int64),
            released_target=np.asarray(base_target, dtype=np.float64),
            variances=np.asarray(base_variances, dtype=np.float64),
            alpha_l2=0.0125,
            alpha_linf=0.0125,
            stage="base",
            round_index=-1,
        )
    ]
    for round_index, (local_target, local_variance, local_indices) in enumerate(
        zip(
            refinement_targets,
            refinement_variances,
            refinement_indices,
            strict=True,
        )
    ):
        streams.append(
            RCECoefficientStream(
                name=f"refinement_{round_index:02d}",
                coefficient_indices=np.asarray(local_indices, dtype=np.int64),
                released_target=np.asarray(local_target, dtype=np.float64),
                variances=np.asarray(local_variance, dtype=np.float64),
                alpha_l2=0.0125 / rounds,
                alpha_linf=0.0125 / rounds,
                stage="refinement",
                round_index=round_index,
            )
        )
    return StreamwiseRCEConfidenceSet(
        canonical_dimension=len(target),
        combined_target=target,
        streams=tuple(streams),
        alpha_total=float(alpha_total),
    )


__all__ = [
    "RCECoefficientStream",
    "STREAMWISE_RCE_METHOD",
    "StreamwiseRCEConfidenceSet",
    "build_cdwf_streamwise_confidence",
]
