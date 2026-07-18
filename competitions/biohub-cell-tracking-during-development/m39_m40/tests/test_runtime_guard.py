"""Unit tests for the M39/M40 runtime guard.

Simulates GREEN, AMBER, RED, FALLBACK, EXPIRED, and the finalization-reserve
gate on ``can_start``. Never sleeps in real time — uses a fake clock.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m39_m40.runtime_guard import (
    PhaseRecord,
    RuntimeBudget,
    RuntimeLevel,
    RuntimePolicy,
)
from m39_m40.schemas import validate_runtime_report


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


DEFAULT_POLICY = RuntimePolicy()          # 90 target, 105 eng, 110 hidden, 120 abs


class RuntimeGuardTests(unittest.TestCase):
    def test_policy_validates(self):
        RuntimePolicy().validate()

    def test_policy_rejects_out_of_order(self):
        with self.assertRaises(ValueError):
            RuntimePolicy(hard_engineering_seconds=1.0).validate()

    def test_policy_rejects_bad_fractions(self):
        with self.assertRaises(ValueError):
            RuntimePolicy(amber_fraction=0.9, red_fraction=0.5).validate()

    def test_starts_at_green(self):
        b = RuntimeBudget(clock=_FakeClock())
        self.assertEqual(b.level(), RuntimeLevel.GREEN)
        self.assertEqual(b.decision()["level"], "GREEN")

    def test_transitions_to_amber(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        c.advance(0.70 * b.policy.hard_engineering_seconds)
        self.assertEqual(b.level(), RuntimeLevel.AMBER)

    def test_transitions_to_red(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        c.advance(0.85 * b.policy.hard_engineering_seconds)
        self.assertEqual(b.level(), RuntimeLevel.RED)

    def test_transitions_to_fallback_by_fraction(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        c.advance(0.94 * b.policy.hard_engineering_seconds)
        self.assertEqual(b.level(), RuntimeLevel.FALLBACK)

    def test_transitions_to_expired(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        c.advance(b.policy.absolute_seconds + 1)
        self.assertEqual(b.level(), RuntimeLevel.EXPIRED)

    def test_can_start_respects_finalization_reserve(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        # 100 minutes gone, only 5 min left before eng cap (105) and 10 min reserve required
        c.advance(100 * 60)
        self.assertFalse(b.can_start(estimated_seconds=1))
        # move backwards is illegal; instead reset with fresh budget
        b2 = RuntimeBudget(clock=_FakeClock())
        self.assertTrue(b2.can_start(estimated_seconds=30 * 60))
        self.assertFalse(b2.can_start(estimated_seconds=100 * 60))

    def test_decision_downgrades_when_next_phase_too_big(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        # plenty of time overall, but next phase estimate + finalization reserve exceeds cap
        d = b.decision(estimated_next_phase_seconds=200 * 60)
        self.assertEqual(d["level"], "FALLBACK")
        self.assertEqual(d["action"], "abort_experiment_validate_and_write_fallback")

    def test_start_phase_raises_when_no_budget(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        c.advance(0.95 * b.policy.hard_engineering_seconds)
        with self.assertRaises(TimeoutError):
            b.start_phase("post_processing", estimated_seconds=30 * 60)

    def test_phase_records_duration(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        b.start_phase("detection", estimated_seconds=60)
        c.advance(45)
        rec = b.end_phase()
        self.assertEqual(rec.name, "detection")
        self.assertAlmostEqual(rec.duration_seconds or 0.0, 45.0, places=3)
        self.assertEqual(rec.status, "completed")

    def test_double_start_is_rejected(self):
        b = RuntimeBudget(clock=_FakeClock())
        b.start_phase("a")
        with self.assertRaises(RuntimeError):
            b.start_phase("b")

    def test_end_without_start_is_rejected(self):
        b = RuntimeBudget(clock=_FakeClock())
        with self.assertRaises(RuntimeError):
            b.end_phase()

    def test_negative_estimates_are_rejected(self):
        b = RuntimeBudget(clock=_FakeClock())
        with self.assertRaises(ValueError):
            b.can_start(estimated_seconds=-1)

    def test_report_shape_and_write(self):
        c = _FakeClock()
        b = RuntimeBudget(clock=c)
        b.start_phase("environment_setup", estimated_seconds=30)
        c.advance(20)
        b.end_phase()
        b.mark_event("checkpoint", note="ok")
        r = b.report()
        v = validate_runtime_report(r)
        self.assertTrue(v["ok"])
        self.assertIn("environment_setup", v["phase_names"])
        with tempfile.TemporaryDirectory() as tmp:
            out = b.write_report(Path(tmp) / "rr.json")
            payload = json.loads(out.read_text())
            self.assertEqual(payload["current_level"], b.level().value)

    def test_deterministic_rerun_produces_identical_reports(self):
        def run_once() -> dict:
            c = _FakeClock()
            b = RuntimeBudget(clock=c)
            for name, dur in (("environment_setup", 30), ("detection", 600)):
                b.start_phase(name, estimated_seconds=dur)
                c.advance(dur)
                b.end_phase()
            return b.report()
        self.assertEqual(run_once(), run_once())


if __name__ == "__main__":
    unittest.main()
