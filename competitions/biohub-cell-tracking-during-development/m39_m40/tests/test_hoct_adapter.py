"""Schema-only tests for the HOCT adapter.

These tests never call real HOCT. They verify:

- HoctAdapterConfig rejects bad max_delta_t / n_neighbors / TTA / long-gap.
- Node & edge schema validators catch missing fields, missing features,
  wrong delta_t, and reversed time order.
- probe_hoct_source degrades to a PENDING/import_failed status when hoct
  is absent and, when a fake stub module is on the path, is classified
  as ``stub_detected_blocked``.
- Every entry point that would need real inference RAISES HoctBlockedError.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from m39_m40.hoct_adapter import (
    DEFAULT_N_NEIGHBORS,
    HOCT_REQUIRED_EDGE_FEATURES,
    HOCT_REQUIRED_NODE_FEATURES,
    HoctAdapterConfig,
    HoctBlockedError,
    HoctPrereqs,
    MAX_DELTA_T,
    MAX_N_NEIGHBORS,
    _looks_like_stub,
    build_hoct_graph,
    check_prereqs,
    probe_hoct_source,
    run_shadow_inference,
    validate_edge_schema,
    validate_node_schema,
)


def _good_node(i: int) -> dict:
    return {
        "node_id": i, "dataset": "d", "t": 0,
        "z_um": 0.0, "y_um": 0.0, "x_um": 0.0,
        "z_vox": 0, "y_vox": 0, "x_vox": 0,
        "detection_score": 0.9, "mask_label": 1,
        "mask_area_vox": 200, "mask_intensity_mean": 100.0,
        "mask_intensity_std": 10.0,
    }


def _good_edge(i: int, dt: int = MAX_DELTA_T) -> dict:
    return {
        "edge_id": i, "dataset": "d",
        "source_id": i, "target_id": i + 1,
        "source_t": 0, "target_t": 0 + dt, "delta_t": dt,
        "displacement_um": 1.0,
        "source_detection_score": 0.9, "target_detection_score": 0.9,
    }


class HoctConfigTests(unittest.TestCase):
    def test_default_config_validates(self):
        HoctAdapterConfig().validate()

    def test_rejects_bad_kind(self):
        with self.assertRaises(ValueError):
            HoctAdapterConfig(kind="bogus").validate()

    def test_pins_max_delta_t(self):
        with self.assertRaises(ValueError):
            HoctAdapterConfig(max_delta_t=2).validate()

    def test_rejects_too_many_neighbors(self):
        with self.assertRaises(ValueError):
            HoctAdapterConfig(n_neighbors=MAX_N_NEIGHBORS + 1).validate()

    def test_rejects_tta(self):
        with self.assertRaises(ValueError):
            HoctAdapterConfig(use_tta=True).validate()

    def test_rejects_long_gap(self):
        with self.assertRaises(ValueError):
            HoctAdapterConfig(use_long_gap_pass=True).validate()


class HoctSchemaTests(unittest.TestCase):
    def test_valid_node_rows_pass(self):
        v = validate_node_schema([_good_node(i) for i in range(3)])
        self.assertTrue(v["ok"], v)

    def test_missing_node_feature_detected(self):
        rows = [_good_node(0)]
        del rows[0]["mask_area_vox"]
        v = validate_node_schema(rows)
        self.assertFalse(v["ok"])
        self.assertIn("mask_area_vox", v["missing_features"])

    def test_empty_input_is_not_ok(self):
        self.assertFalse(validate_node_schema([])["ok"])

    def test_valid_edge_rows_pass(self):
        v = validate_edge_schema([_good_edge(0), _good_edge(1)])
        self.assertTrue(v["ok"], v)

    def test_wrong_delta_t_rejected(self):
        v = validate_edge_schema([_good_edge(0, dt=2)])
        self.assertFalse(v["delta_t_ok"])
        self.assertFalse(v["ok"])

    def test_reversed_time_order_rejected(self):
        row = _good_edge(0)
        row["source_t"], row["target_t"] = 5, 3
        row["delta_t"] = -2
        v = validate_edge_schema([row])
        self.assertFalse(v["time_order_ok"])
        self.assertFalse(v["ok"])

    def test_missing_edge_feature_detected(self):
        row = _good_edge(0)
        del row["displacement_um"]
        v = validate_edge_schema([row])
        self.assertFalse(v["ok"])
        self.assertIn("displacement_um", v["missing_features"])


class HoctProbeTests(unittest.TestCase):
    def test_absent_hoct_reports_pending(self):
        # This sandbox has no hoct
        p = probe_hoct_source()
        self.assertFalse(p.get("hoct_installed"))
        self.assertEqual(p.get("create_graph_from_points_status"), "PENDING_REAL_ENV")

    def test_stub_heuristic_direct(self):
        self.assertTrue(_looks_like_stub("def f():\n    pass\n"))
        self.assertTrue(_looks_like_stub("def f():\n    raise NotImplementedError\n"))
        self.assertTrue(_looks_like_stub("def f():\n    return None\n"))
        self.assertFalse(_looks_like_stub(
            "def f():\n    g = build_graph()\n    return g.finalize()\n"
        ))

    def test_probe_classifies_installed_stub_as_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "hoct"
            (pkg / "graph").mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "graph" / "__init__.py").write_text(textwrap.dedent("""
                def create_graph_from_points(points):
                    raise NotImplementedError
            """).lstrip())
            sys.path.insert(0, tmp)
            for mod in ("hoct", "hoct.graph"):
                sys.modules.pop(mod, None)
            try:
                p = probe_hoct_source()
                self.assertTrue(p["hoct_installed"])
                self.assertEqual(p["create_graph_from_points_status"],
                                 "stub_detected_blocked")
            finally:
                for mod in ("hoct", "hoct.graph"):
                    sys.modules.pop(mod, None)
                sys.path.remove(tmp)


class HoctPrereqsTests(unittest.TestCase):
    def test_missing_prereqs_reported(self):
        p = check_prereqs(weight_path=None, images_dir=None, baseline_detections_path=None)
        for key in ("torch", "hoct_create_graph_from_points_or_valid_adapter",
                    "hoct_weight_file", "images_dir", "baseline_detections"):
            self.assertIn(key, p.missing)


class HoctBlockedTests(unittest.TestCase):
    def test_build_hoct_graph_blocks_without_prereqs(self):
        cfg = HoctAdapterConfig()
        pf = check_prereqs(weight_path=None, images_dir=None, baseline_detections_path=None)
        with self.assertRaises(HoctBlockedError):
            build_hoct_graph(detections=None, config=cfg, prereqs=pf)

    def test_run_shadow_inference_is_blocked(self):
        with self.assertRaises(HoctBlockedError):
            run_shadow_inference()


if __name__ == "__main__":
    unittest.main()
