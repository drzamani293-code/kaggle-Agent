"""Wall-clock runtime governor for BIOHUB Kaggle notebooks.

Designed to preserve enough time to validate and write submission.csv before a
two-hour notebook ceiling. It does not kill processes by itself; callers must
check `decision()` before expensive phases and activate the requested
degradation/fallback path.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RuntimeLevel(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    FALLBACK = "FALLBACK"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class RuntimePolicy:
    # Conservative engineering values, all in seconds.
    target_seconds: float = 90 * 60
    hard_engineering_seconds: float = 105 * 60
    hidden_safety_seconds: float = 110 * 60
    absolute_seconds: float = 120 * 60
    finalization_reserve_seconds: float = 10 * 60

    amber_fraction: float = 0.68
    red_fraction: float = 0.82
    fallback_fraction: float = 0.91

    def validate(self) -> None:
        values = (
            self.target_seconds,
            self.hard_engineering_seconds,
            self.hidden_safety_seconds,
            self.absolute_seconds,
            self.finalization_reserve_seconds,
        )
        if any(v <= 0 for v in values):
            raise ValueError("All runtime policy durations must be positive.")
        if not (
            self.target_seconds
            < self.hard_engineering_seconds
            <= self.hidden_safety_seconds
            < self.absolute_seconds
        ):
            raise ValueError("Runtime limits must be strictly ordered.")
        if not (0 < self.amber_fraction < self.red_fraction < self.fallback_fraction < 1):
            raise ValueError("Degradation fractions must be strictly ordered in (0, 1).")


@dataclass
class PhaseRecord:
    name: str
    start_elapsed_seconds: float
    end_elapsed_seconds: float | None = None
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.end_elapsed_seconds is None:
            return None
        return self.end_elapsed_seconds - self.start_elapsed_seconds


class RuntimeBudget:
    def __init__(
        self,
        policy: RuntimePolicy | None = None,
        *,
        clock=time.monotonic,
    ) -> None:
        self.policy = policy or RuntimePolicy()
        self.policy.validate()
        self._clock = clock
        self._started = clock()
        self._phases: list[PhaseRecord] = []
        self._active: PhaseRecord | None = None
        self.events: list[dict[str, Any]] = []

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    def remaining_to_engineering_cap(self) -> float:
        return max(0.0, self.policy.hard_engineering_seconds - self.elapsed_seconds())

    def remaining_to_absolute_cap(self) -> float:
        return max(0.0, self.policy.absolute_seconds - self.elapsed_seconds())

    def experimental_seconds_available(self) -> float:
        """Time available while still preserving finalization reserve."""
        return max(
            0.0,
            self.policy.hard_engineering_seconds
            - self.policy.finalization_reserve_seconds
            - self.elapsed_seconds(),
        )

    def level(self) -> RuntimeLevel:
        elapsed = self.elapsed_seconds()
        cap = self.policy.hard_engineering_seconds
        if elapsed >= self.policy.absolute_seconds:
            return RuntimeLevel.EXPIRED
        if elapsed >= self.policy.fallback_fraction * cap:
            return RuntimeLevel.FALLBACK
        if elapsed >= self.policy.red_fraction * cap:
            return RuntimeLevel.RED
        if elapsed >= self.policy.amber_fraction * cap:
            return RuntimeLevel.AMBER
        return RuntimeLevel.GREEN

    def can_start(
        self,
        *,
        estimated_seconds: float,
        extra_reserve_seconds: float = 0.0,
    ) -> bool:
        if estimated_seconds < 0 or extra_reserve_seconds < 0:
            raise ValueError("Estimated duration and reserve must be non-negative.")
        needed = (
            estimated_seconds
            + self.policy.finalization_reserve_seconds
            + extra_reserve_seconds
        )
        return self.remaining_to_engineering_cap() >= needed

    def decision(self, *, estimated_next_phase_seconds: float = 0.0) -> dict[str, Any]:
        level = self.level()
        action = {
            RuntimeLevel.GREEN: "run_full_selected_plan",
            RuntimeLevel.AMBER: "disable_tta_and_optional_diagnostics",
            RuntimeLevel.RED: "disable_long_gap_reduce_neighbors_cap_ilp",
            RuntimeLevel.FALLBACK: "abort_experiment_validate_and_write_fallback",
            RuntimeLevel.EXPIRED: "stop_immediately",
        }[level]
        if not self.can_start(estimated_seconds=estimated_next_phase_seconds):
            level = RuntimeLevel.FALLBACK
            action = "abort_experiment_validate_and_write_fallback"
        result = {
            "level": level.value,
            "action": action,
            "elapsed_seconds": self.elapsed_seconds(),
            "remaining_engineering_seconds": self.remaining_to_engineering_cap(),
            "remaining_absolute_seconds": self.remaining_to_absolute_cap(),
            "experimental_seconds_available": self.experimental_seconds_available(),
            "estimated_next_phase_seconds": estimated_next_phase_seconds,
        }
        self.events.append({"type": "decision", **result})
        return result

    def start_phase(
        self,
        name: str,
        *,
        estimated_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._active is not None:
            raise RuntimeError(f"Phase {self._active.name!r} is still active.")
        decision = self.decision(estimated_next_phase_seconds=estimated_seconds)
        if decision["level"] in {RuntimeLevel.FALLBACK.value, RuntimeLevel.EXPIRED.value}:
            raise TimeoutError(
                f"Not enough safe budget to start phase {name!r}: {decision}"
            )
        self._active = PhaseRecord(
            name=name,
            start_elapsed_seconds=self.elapsed_seconds(),
            metadata=metadata or {},
        )
        self._phases.append(self._active)
        return decision

    def end_phase(
        self,
        *,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> PhaseRecord:
        if self._active is None:
            raise RuntimeError("No active phase.")
        self._active.end_elapsed_seconds = self.elapsed_seconds()
        self._active.status = status
        if metadata:
            self._active.metadata.update(metadata)
        completed = self._active
        self._active = None
        return completed

    def mark_event(self, name: str, **metadata: Any) -> None:
        self.events.append(
            {
                "type": "event",
                "name": name,
                "elapsed_seconds": self.elapsed_seconds(),
                **metadata,
            }
        )

    def report(self) -> dict[str, Any]:
        return {
            "policy": asdict(self.policy),
            "elapsed_seconds": self.elapsed_seconds(),
            "current_level": self.level().value,
            "active_phase": self._active.name if self._active else None,
            "phases": [
                {
                    **asdict(phase),
                    "duration_seconds": phase.duration_seconds,
                }
                for phase in self._phases
            ],
            "events": self.events,
        }

    def write_report(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output


if __name__ == "__main__":
    # Tiny smoke test without waiting in real time.
    class FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        def advance(self, seconds: float) -> None:
            self.t += seconds

    fake = FakeClock()
    budget = RuntimeBudget(clock=fake)
    assert budget.level() == RuntimeLevel.GREEN
    budget.start_phase("setup", estimated_seconds=60)
    fake.advance(60)
    budget.end_phase()
    fake.advance(0.70 * budget.policy.hard_engineering_seconds)
    assert budget.level() in {RuntimeLevel.AMBER, RuntimeLevel.RED}
    report_path = budget.write_report("runtime_report_smoke.json")
    print(report_path)
