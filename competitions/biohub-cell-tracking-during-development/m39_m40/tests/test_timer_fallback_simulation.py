"""End-to-end timer/fallback simulation.

Uses an artificially tiny 60-second budget with proportional AMBER/RED/
FALLBACK fractions to walk through the full GREEN -> AMBER -> RED ->
FALLBACK ladder, then confirms that ``kaggle_runner.run`` still returns a
fallback-only response instead of raising."""

from __future__ import annotations

import unittest
from pathlib import Path

from m39_m40.kaggle_runner import build_default_budget, run
from m39_m40.runtime_guard import RuntimeBudget, RuntimeLevel, RuntimePolicy


REPO_ARCHIVE = Path(__file__).resolve().parents[2] / "archive" / "m19c"


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


TINY = RuntimePolicy(
    target_seconds=30,
    hard_engineering_seconds=60,
    hidden_safety_seconds=65,
    absolute_seconds=75,
    finalization_reserve_seconds=6,
)


class TinyBudgetLadderTests(unittest.TestCase):
    def test_full_degradation_ladder_walks_correctly(self):
        c = _FakeClock()
        b = RuntimeBudget(policy=TINY, clock=c)
        # start GREEN
        self.assertEqual(b.level(), RuntimeLevel.GREEN)
        # advance to AMBER (0.68 * 60 = 40.8s)
        c.advance(41)
        self.assertEqual(b.level(), RuntimeLevel.AMBER)
        # advance to RED (0.82 * 60 = 49.2s)
        c.advance(10)          # now at 51
        self.assertEqual(b.level(), RuntimeLevel.RED)
        # advance to FALLBACK (0.91 * 60 = 54.6s)
        c.advance(5)           # now at 56
        self.assertEqual(b.level(), RuntimeLevel.FALLBACK)
        # advance past absolute (75s)
        c.advance(30)
        self.assertEqual(b.level(), RuntimeLevel.EXPIRED)

    def test_finalization_reserve_blocks_late_start(self):
        c = _FakeClock()
        b = RuntimeBudget(policy=TINY, clock=c)
        # 55s in of 60s cap, 6s reserve -> only ~-1s available; anything blocks
        c.advance(55)
        self.assertFalse(b.can_start(estimated_seconds=1))
        with self.assertRaises(TimeoutError):
            b.start_phase("ilp_graph_solve", estimated_seconds=1)

    def test_experimental_seconds_available_never_negative(self):
        c = _FakeClock()
        b = RuntimeBudget(policy=TINY, clock=c)
        c.advance(200)
        self.assertGreaterEqual(b.experimental_seconds_available(), 0.0)
        self.assertGreaterEqual(b.remaining_to_engineering_cap(), 0.0)


class RunFallbackReturnsWithoutRaise(unittest.TestCase):
    def test_run_never_raises_in_sandbox(self):
        # No mount, no CUDA -> FALLBACK_ONLY. Must not raise.
        r = run(archive_dir=REPO_ARCHIVE)
        self.assertEqual(r["mode"], "FALLBACK_ONLY")
        # Runtime report always returned, even when experimental path is skipped
        rr = r["runtime_report"]
        self.assertIn("policy", rr)
        self.assertIn("events", rr)


if __name__ == "__main__":
    unittest.main()
