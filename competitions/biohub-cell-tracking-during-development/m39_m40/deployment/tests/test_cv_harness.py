"""Unit tests for cv_harness."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from m39_m40 import PENDING_REAL_ENV
from m39_m40.deployment.cv_harness import (
    aggregate,
    check_metric_env,
    ensure_metric_on_path,
)


def _make_stub_bundle_with_metric(root: Path) -> None:
    """Create a bundle whose vendor/royerlab_cellmot/src/tracking_cellmot
    package exists, so ensure_metric_on_path returns True."""
    src = root / "vendor" / "royerlab_cellmot" / "src" / "tracking_cellmot"
    src.mkdir(parents=True, exist_ok=True)
    (src / "__init__.py").write_text("__version__ = '0.0.0-test'\n")
    (src / "metrics.py").write_text("def evaluate(*a, **k): return {}\n")


class EnsureMetricOnPathTests(unittest.TestCase):
    def test_missing_package_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(ensure_metric_on_path(Path(td)))

    def test_present_package_is_added_to_sys_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_stub_bundle_with_metric(root)
            self.assertTrue(ensure_metric_on_path(root))
            self.assertIn(
                str(root / "vendor" / "royerlab_cellmot" / "src"),
                sys.path,
            )
            # cleanup: remove the injected entry (avoid leaking to sibling tests)
            sys.path.remove(str(root / "vendor" / "royerlab_cellmot" / "src"))
            # also drop any imported test stub so it doesn't shadow the real
            # tracking_cellmot in other tests.
            for mod in ("tracking_cellmot", "tracking_cellmot.metrics"):
                sys.modules.pop(mod, None)


class CheckMetricEnvTests(unittest.TestCase):
    def test_verdict_missing_when_package_absent(self):
        with tempfile.TemporaryDirectory() as td:
            report = check_metric_env(Path(td))
            self.assertEqual(report.verdict, "OFFICIAL_METRIC_IMPORT_FAILED")

    def test_deps_reported_when_package_present(self):
        # We don't force tracksdata/polars/geff to be installed; the report
        # just reflects reality (probably OFFICIAL_METRIC_DEPENDENCY_MISSING
        # or OFFICIAL_METRIC_IMPORT_FAILED in this sandbox).
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_stub_bundle_with_metric(root)
            report = check_metric_env(root)
            self.assertIn(report.verdict,
                          {"OK", "OFFICIAL_METRIC_DEPENDENCY_MISSING",
                           "OFFICIAL_METRIC_IMPORT_FAILED"})
            # cleanup as above
            sys.path.remove(str(root / "vendor" / "royerlab_cellmot" / "src"))
            for mod in list(sys.modules):
                if mod == "tracking_cellmot" or mod.startswith("tracking_cellmot."):
                    sys.modules.pop(mod, None)


class AggregateTests(unittest.TestCase):
    def _row(self, dataset: str, config_id: str, score: float) -> dict:
        return {
            "dataset": dataset, "config_id": config_id,
            "adjusted_edge_jaccard": score, "division_tp": 0, "division_fp": 0,
            "division_fn": 0, "division_jaccard": 0.0, "node_count_penalty": 0.0,
            "n_nodes": 100, "n_edges": 200, "n_divisions": 0,
            "runtime_seconds": 12.5, "peak_memory_mb": 8000.0,
            "official_score": score, "status": "computed",
        }

    def test_pending_when_audits_not_ok(self):
        rows = [self._row("d1", "c1", 0.9)]
        s = aggregate(rows=rows, official_metric_audit_ok=False, split_audit_ok=True)
        self.assertEqual(s["status"], PENDING_REAL_ENV)
        s = aggregate(rows=rows, official_metric_audit_ok=True, split_audit_ok=False)
        self.assertEqual(s["status"], PENDING_REAL_ENV)

    def test_final_when_all_gates_pass(self):
        rows = [self._row("d1", "c1", 0.8), self._row("d2", "c1", 0.9)]
        s = aggregate(rows=rows, official_metric_audit_ok=True, split_audit_ok=True)
        self.assertEqual(s["status"], "final")
        self.assertAlmostEqual(s["macro_official_score"], 0.85)
        self.assertEqual(s["winner_config_id"], "c1")

    def test_no_computed_rows_stays_pending(self):
        # only PENDING rows -> nothing to summarise
        s = aggregate(rows=[], official_metric_audit_ok=True, split_audit_ok=True)
        self.assertEqual(s["status"], PENDING_REAL_ENV)


if __name__ == "__main__":
    unittest.main()
