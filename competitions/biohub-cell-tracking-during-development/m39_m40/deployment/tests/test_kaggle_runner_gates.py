"""Unit tests for kaggle_one_cell_runner gates (all G-failures ->
fallback, all gates pass -> EXPERIMENTAL_SUBMITTED)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from m39_m40.deployment import kaggle_one_cell_runner as runner
from m39_m40.deployment import baseline_association as ba
from m39_m40.deployment import hoct_association as ha
from m39_m40.deployment.baseline_association import BaselineAssociationResult
from m39_m40.deployment.hoct_association import HoctAssociationResult
from m39_m40.hoct_adapter import HoctAdapterConfig


class DiscoveryTests(unittest.TestCase):
    def test_finds_marker_by_walking(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "kaggle_input"
            bundle = root / "datasets" / "user" / "slug"
            bundle.mkdir(parents=True)
            (bundle / "M39_M40_BUNDLE.marker").write_text("hi\n")
            found = runner.discover_bundle_root(hints=(str(root),))
            self.assertEqual(found, bundle)

    def test_returns_none_when_no_marker(self):
        with tempfile.TemporaryDirectory() as td:
            found = runner.discover_bundle_root(hints=(td,))
            self.assertIsNone(found)


class GateFailureTests(unittest.TestCase):
    def setUp(self):
        ba.clear()
        ha.clear()

    def tearDown(self):
        ba.clear()
        ha.clear()

    def test_g1_bundle_missing_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "submission.csv"
            result = runner.main(bundle_root=None, output_path=out)
        self.assertEqual(result.mode, "FALLBACK_M19C")
        self.assertFalse(result.gates[0].passed)
        self.assertEqual(result.gates[0].name, "G1_bundle_manifest")

    def test_g1_manifest_missing_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "bundle"
            bundle.mkdir()
            (bundle / "M39_M40_BUNDLE.marker").write_text("x\n")
            # no BUNDLE_MANIFEST.json -> load_manifest returns error dict
            result = runner.main(bundle_root=bundle,
                                 output_path=Path(td) / "submission.csv")
        self.assertEqual(result.mode, "FALLBACK_M19C")
        first_failed = next(g for g in result.gates if not g.passed)
        self.assertEqual(first_failed.name, "G1_bundle_manifest")


class BaselineHoctContractsTests(unittest.TestCase):
    def setUp(self):
        ba.clear()
        ha.clear()

    def tearDown(self):
        ba.clear()
        ha.clear()

    def test_baseline_registry_roundtrip(self):
        def solver(req):
            return BaselineAssociationResult(
                dataset=req.dataset, config_id=req.config_id,
                pred_graph={"n": 3}, n_nodes=3, n_edges=2,
                runtime_seconds=0.1, peak_memory_mb=1.0,
                detections_reused=True,
            )
        ba.register("x", solver)
        self.assertTrue(ba.has("x"))
        self.assertIn("x", ba.list_registered())
        got = ba.get("x")
        r = got(ba.BaselineAssociationRequest(dataset="d", config_id="c",
                                              cached_detections=None,
                                              dataset_manifest={}))
        self.assertEqual(r.n_nodes, 3)

    def test_hoct_identity_broken_refused(self):
        def bad_solver(req):
            return HoctAssociationResult(
                dataset=req.dataset, config_id=req.config_id,
                pred_graph={}, n_nodes=1, n_edges=0,
                runtime_seconds=0.0, peak_memory_mb=1.0,
                detections_reused=False,               # <— broken
                hoct_config_used=req.hoct_config,
            )
        ha.register("bad", bad_solver)
        req = ha.HoctAssociationRequest(
            dataset="d", config_id="c",
            cached_detections="X", dataset_manifest={},
            hoct_config=HoctAdapterConfig(),
        )
        with self.assertRaises(ha.HoctIdentityBrokenError):
            ha.run(name="bad", request=req)

    def test_hoct_config_deviation_refused(self):
        def deviant(req):
            other = HoctAdapterConfig(max_delta_t=1, n_neighbors=5,
                                      use_tta=False, use_long_gap_pass=False)
            return HoctAssociationResult(
                dataset=req.dataset, config_id=req.config_id,
                pred_graph={}, n_nodes=1, n_edges=0,
                runtime_seconds=0.0, peak_memory_mb=1.0,
                detections_reused=True,
                hoct_config_used=other,
            )
        ha.register("deviant", deviant)
        req = ha.HoctAssociationRequest(
            dataset="d", config_id="c",
            cached_detections="X", dataset_manifest={},
            hoct_config=HoctAdapterConfig(max_delta_t=1, n_neighbors=3,
                                          use_tta=False, use_long_gap_pass=False),
        )
        from m39_m40.hoct_adapter import HoctBlockedError
        with self.assertRaises(HoctBlockedError):
            ha.run(name="deviant", request=req)

    def test_hoct_config_pinned_accepted(self):
        cfg = HoctAdapterConfig(max_delta_t=1, n_neighbors=3,
                                use_tta=False, use_long_gap_pass=False)

        def good(req):
            return HoctAssociationResult(
                dataset=req.dataset, config_id=req.config_id,
                pred_graph={}, n_nodes=1, n_edges=0,
                runtime_seconds=0.0, peak_memory_mb=1.0,
                detections_reused=True, hoct_config_used=req.hoct_config,
            )
        ha.register("good", good)
        req = ha.HoctAssociationRequest(
            dataset="d", config_id="c",
            cached_detections="X", dataset_manifest={}, hoct_config=cfg,
        )
        r = ha.run(name="good", request=req)
        self.assertTrue(r.detections_reused)


if __name__ == "__main__":
    unittest.main()
