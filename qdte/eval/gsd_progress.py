from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GSDProgressObserver:
    n_synthetic: int
    started_at: float
    min_generation: int
    relative_improvement_threshold: float
    plateau_checkpoints: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    consecutive_plateau_checkpoints: int = 0
    stop_generation: int | None = None

    @staticmethod
    def _relative_improvement(current: float, previous: float) -> float:
        if not np.isfinite(current) or not np.isfinite(previous):
            return float("inf")
        return max(0.0, (previous - current) / max(abs(previous), 1.0e-30))

    def record(
        self,
        generation: int,
        best_fitness: Any,
        previous_fitness: Any,
        strategy_weights: Any,
        loop_time_seconds: float,
    ) -> None:
        current = float(np.asarray(best_fitness))
        previous = float(np.asarray(previous_fitness))
        weights = np.asarray(strategy_weights, dtype=np.float64).reshape(-1)
        self.rows.append(
            {
                "generation": int(generation),
                "wall_time_seconds": float(time.time() - self.started_at),
                "loop_time_seconds": float(loop_time_seconds),
                "rate_l2_squared": current,
                "target_count_loss": float(0.5 * (self.n_synthetic**2) * current),
                "relative_improvement_from_previous_checkpoint": self._relative_improvement(
                    current,
                    previous,
                ),
                "consecutive_plateau_checkpoints": int(self.consecutive_plateau_checkpoints),
                "strategy_weights": json.dumps(weights.astype(float).tolist()),
                "checkpoint_kind": "periodic",
            }
        )

    def should_stop(
        self,
        generation: int,
        best_fitness: Any,
        previous_fitness: Any,
    ) -> bool:
        generation_i = int(generation)
        if self.plateau_checkpoints <= 0 or generation_i < self.min_generation:
            return False
        relative_improvement = self._relative_improvement(
            float(np.asarray(best_fitness)),
            float(np.asarray(previous_fitness)),
        )
        if relative_improvement < self.relative_improvement_threshold:
            self.consecutive_plateau_checkpoints += 1
        else:
            self.consecutive_plateau_checkpoints = 0
        if self.rows and int(self.rows[-1]["generation"]) == generation_i:
            self.rows[-1]["consecutive_plateau_checkpoints"] = int(
                self.consecutive_plateau_checkpoints
            )
        if self.consecutive_plateau_checkpoints >= self.plateau_checkpoints:
            self.stop_generation = generation_i + 1
            return True
        return False

    def append_final(
        self,
        *,
        actual_generations: int,
        rate_l2_squared: float,
        target_count_loss: float,
    ) -> None:
        self.rows.append(
            {
                "generation": int(actual_generations),
                "wall_time_seconds": float(time.time() - self.started_at),
                "loop_time_seconds": None,
                "rate_l2_squared": float(rate_l2_squared),
                "target_count_loss": float(target_count_loss),
                "relative_improvement_from_previous_checkpoint": None,
                "consecutive_plateau_checkpoints": int(self.consecutive_plateau_checkpoints),
                "strategy_weights": "[]",
                "checkpoint_kind": "final",
            }
        )

    def write_csv(self, path: Path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False)
