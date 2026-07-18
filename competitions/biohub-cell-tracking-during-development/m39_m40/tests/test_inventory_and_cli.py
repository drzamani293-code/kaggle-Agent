"""Inventory generator, metric-vendor scaffold, split-audit scaffold, and
offline-import scaffold tests. All CLI entry points are exercised as
Python functions to avoid touching sys.argv globally."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m39_m40 import PENDING_REAL_ENV
from m39_m40 import inventory, metric_vendor, offline_check, split_audit


COMP_DIR = Path(__file__).resolve().parents[1].parent


class InventoryTests(unittest.TestCase):
    def test_build_inventory_shape(self):
        inv = inventory.build_inventory(COMP_DIR)
        self.assertEqual(inv["milestone"], "M39")
        self.assertIn("git", inv)
        self.assertIn("env", inv)
        self.assertIn("m19c_archive", inv)
        self.assertGreater(inv["counts"]["one_cell_txt"], 0)
        self.assertGreater(inv["counts"]["runners_py"], 0)
        self.assertGreater(inv["counts"]["notebooks"], 0)
        self.assertTrue(inv["m19c_archive"]["manifest_present"])
        self.assertTrue(inv["m19c_archive"]["manifest_ok"])
        # env: honest reporting only
        self.assertIn(inv["env"]["has_torch"], (True, False))
        self.assertIn(inv["env"]["cuda_available"], (True, False))

    def test_write_inventory_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "inv.json"
            inv = inventory.build_inventory(COMP_DIR)
            inventory.write_inventory(inv, out)
            loaded = json.loads(out.read_text())
            self.assertEqual(loaded["milestone"], "M39")


class MetricVendorTests(unittest.TestCase):
    def test_pending_when_no_vendor_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = metric_vendor.audit(Path(tmp))
            self.assertEqual(d["status"], PENDING_REAL_ENV)
            self.assertFalse(d["license_present"])
            self.assertEqual(d["match_distance_micrometres"], 7.0)
            self.assertEqual(d["voxel_scale_z"], 1.625)

    def test_local_metric_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vroot = Path(tmp) / "reference" / "official_metric"
            vroot.mkdir(parents=True)
            (vroot / "local_metric.py").write_text("pass\n")
            (vroot / "LICENSE").write_text("MIT\n")
            d = metric_vendor.audit(Path(tmp))
            self.assertEqual(d["status"], "REJECTED")
            self.assertFalse(d["silent_fallback_to_local_metric_blocked"])


class SplitAuditTests(unittest.TestCase):
    def test_pending_when_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "membership.csv"
            d = split_audit.audit(Path(tmp), out)
            self.assertEqual(d["status"], PENDING_REAL_ENV)
            self.assertFalse(d["reconstruction_from_seed"])
            self.assertFalse(out.exists())        # never write on empty audit

    def test_verified_when_explicit_split_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference"
            ref.mkdir()
            (ref / "kaggle_test_splits_50ep.json").write_text(
                json.dumps([{"split": 0, "train": ["a", "b"], "test": ["c", "d"]}])
            )
            out = Path(tmp) / "artifacts" / "m39_split_membership.csv"
            d = split_audit.audit(Path(tmp), out)
            self.assertEqual(d["status"], "verified")
            self.assertEqual(d["declared_split_name"], "split_0")
            self.assertEqual(d["train_datasets"], ["a", "b"])
            self.assertEqual(d["validation_datasets"], ["c", "d"])
            self.assertEqual(d["train_validation_overlap"], [])
            self.assertTrue(out.exists())

    def test_leakage_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference"; ref.mkdir()
            (ref / "kaggle_test_splits_50ep.json").write_text(
                json.dumps([{"split": 0, "train": ["a"], "test": ["a", "b"]}])
            )
            out = Path(tmp) / "membership.csv"
            d = split_audit.audit(Path(tmp), out)
            self.assertEqual(d["train_validation_overlap"], ["a"])


class OfflineCheckTests(unittest.TestCase):
    def test_report_shape(self):
        r = offline_check.report()
        self.assertIn("checked", r)
        self.assertIn("importable", r)
        self.assertIn("missing", r)
        self.assertEqual(len(r["results"]), len(r["checked"]))

    def test_missing_module_reported_honestly(self):
        r = offline_check.report(["definitely_not_a_real_module_xyz"])
        self.assertEqual(r["importable"], [])
        self.assertEqual(r["missing"], ["definitely_not_a_real_module_xyz"])


if __name__ == "__main__":
    unittest.main()
