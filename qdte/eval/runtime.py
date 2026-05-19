from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RuntimeStats:
    start_time: float = field(default_factory=time.perf_counter)
    time_measurement_seconds: float = 0.0
    time_init_seconds: float = 0.0
    time_generation_seconds: float = 0.0
    time_candidate_generation_seconds: float = 0.0
    time_scoring_seconds: float = 0.0
    time_transport_seconds: float = 0.0
    time_full_recompute_seconds: float = 0.0
    num_iterations: int = 0
    num_candidates_scored: int = 0
    num_accepted_edits: int = 0
    num_candidates_requested: int = 0
    num_candidate_shortfall: int = 0
    num_directed_candidates: int = 0
    num_random_candidates: int = 0
    num_source_filter_attempts: int = 0
    num_source_filter_failures: int = 0

    def as_dict(self) -> dict[str, float | int]:
        total = time.perf_counter() - self.start_time
        accepted_rate = self.num_accepted_edits / max(1, self.num_candidates_scored)
        scoring_throughput = self.num_candidates_scored / max(1.0e-9, self.time_scoring_seconds)
        source_failure_rate = self.num_source_filter_failures / max(1, self.num_source_filter_attempts)
        return {
            "wall_clock_seconds": float(total),
            "time_measurement_seconds": float(self.time_measurement_seconds),
            "time_init_seconds": float(self.time_init_seconds),
            "time_generation_seconds": float(self.time_generation_seconds),
            "time_candidate_generation_seconds": float(self.time_candidate_generation_seconds),
            "time_scoring_seconds": float(self.time_scoring_seconds),
            "time_transport_seconds": float(self.time_transport_seconds),
            "time_full_recompute_seconds": float(self.time_full_recompute_seconds),
            "num_iterations": int(self.num_iterations),
            "num_candidates_scored": int(self.num_candidates_scored),
            "num_accepted_edits": int(self.num_accepted_edits),
            "num_candidates_requested": int(self.num_candidates_requested),
            "num_candidate_shortfall": int(self.num_candidate_shortfall),
            "num_directed_candidates": int(self.num_directed_candidates),
            "num_random_candidates": int(self.num_random_candidates),
            "num_source_filter_attempts": int(self.num_source_filter_attempts),
            "num_source_filter_failures": int(self.num_source_filter_failures),
            "accepted_rate": float(accepted_rate),
            "candidate_scoring_throughput_per_second": float(scoring_throughput),
            "source_filter_failure_rate": float(source_failure_rate),
        }
