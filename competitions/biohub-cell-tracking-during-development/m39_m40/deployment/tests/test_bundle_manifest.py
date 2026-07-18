"""Unit tests for bundle_manifest + bundle_builder."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from m39_m40.deployment import BUNDLE_MANIFEST_FILENAME, BUNDLE_MARKER_FILENAME
from m39_m40.deployment.bundle_builder import build
from m39_m40.deployment.bundle_manifest import (
    ALL_SLOTS,
    SLOT_HOCT_SOURCE,
    SLOT_HOCT_WEIGHTS,
    SLOT_M19C,
    SLOT_M39_M40,
    SLOT_VENDOR_METRIC,
    STATUS_MISSING,
    STATUS_PRESENT,
    sha256_file,
    verify_manifest,
    load_manifest,
)


COMP_DIR = Path(__file__).resolve().parents[3]


def _write_file(p: Path, body: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)


class BundleBuilderTests(unittest.TestCase):
    def test_refuses_when_hoct_missing_and_not_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bundle"
            result = build(
                comp_dir=COMP_DIR,
                bundle_root=out,
                hoct_source=None,
                hoct_weights=None,
                allow_hoct_missing=False,
                marker_note="test",
            )
        self.assertEqual(result["verdict"], "REFUSED_HOCT_MISSING")
        # required slots still copied
        self.assertGreater(result["file_counts"][SLOT_VENDOR_METRIC], 0)
        self.assertGreater(result["file_counts"][SLOT_M19C], 0)
        self.assertGreater(result["file_counts"][SLOT_M39_M40], 0)

    def test_partial_bundle_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bundle"
            r = build(comp_dir=COMP_DIR, bundle_root=out,
                      hoct_source=None, hoct_weights=None,
                      allow_hoct_missing=True, marker_note="partial")
            self.assertEqual(r["verdict"], "PARTIAL_HOCT_MISSING")
            manifest = load_manifest(out, BUNDLE_MANIFEST_FILENAME)
            slots = manifest["slots"]
            self.assertEqual(slots[SLOT_HOCT_SOURCE]["status"], STATUS_MISSING)
            self.assertEqual(slots[SLOT_HOCT_WEIGHTS]["status"], STATUS_MISSING)

    def test_complete_bundle_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            # Fabricate a synthetic HOCT source + weights so the builder can
            # produce a COMPLETE bundle.  These are TEST-ONLY placeholders
            # meant to exercise the manifest logic — real HOCT source must
            # come from the official MIT-licensed repo at build time.
            hoct_src = Path(td) / "hoct_src"
            _write_file(hoct_src / "hoct" / "__init__.py", b"# stub for tests\n")
            _write_file(hoct_src / "hoct" / "create_graph_from_points.py",
                        b"def create_graph_from_points(*args, **kw):\n"
                        b"    return {'nodes': list(args), 'edges': []}\n")
            _write_file(hoct_src / "PROVENANCE.json",
                        json.dumps({"upstream_repository": "https://example/hoct",
                                    "upstream_commit": "deadbeef" * 5,
                                    "license": "MIT"}, indent=2).encode())
            hoct_wts = Path(td) / "weights"
            _write_file(hoct_wts / "general_v0.pth", b"\x00" * 512)
            out = Path(td) / "bundle"
            r = build(comp_dir=COMP_DIR, bundle_root=out,
                      hoct_source=hoct_src, hoct_weights=hoct_wts,
                      allow_hoct_missing=False, marker_note="complete")
            self.assertEqual(r["verdict"], "COMPLETE")
            self.assertTrue((out / BUNDLE_MARKER_FILENAME).is_file())
            manifest = load_manifest(out, BUNDLE_MANIFEST_FILENAME)
            v = verify_manifest(out, manifest)
            self.assertEqual(v["verdict"], "VERIFIED")

    def test_tamper_flips_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            hoct_src = Path(td) / "hoct_src"
            _write_file(hoct_src / "hoct" / "__init__.py", b"# stub\n")
            _write_file(hoct_src / "hoct" / "create_graph_from_points.py",
                        b"def create_graph_from_points(*a, **k):\n    return {}\n")
            hoct_wts = Path(td) / "weights"
            _write_file(hoct_wts / "general_v0.pth", b"\x00" * 8)
            out = Path(td) / "bundle"
            build(comp_dir=COMP_DIR, bundle_root=out,
                  hoct_source=hoct_src, hoct_weights=hoct_wts,
                  allow_hoct_missing=False, marker_note="")
            # tamper: rewrite one metric file with a different body
            tampered = out / "vendor" / "royerlab_cellmot" / "src" / "tracking_cellmot" / "__init__.py"
            self.assertTrue(tampered.is_file())
            tampered.write_bytes(b"# tampered\n")
            manifest = load_manifest(out, BUNDLE_MANIFEST_FILENAME)
            v = verify_manifest(out, manifest)
            self.assertEqual(v["verdict"], "TAMPERED")
            self.assertTrue(v["mismatches"])

    def test_vendor_metric_hashes_match_provenance(self):
        # Cross-check: the vendored PROVENANCE inside the built bundle
        # continues to match SHA256s of the actual copied files. This is a
        # smoke test that the copy step preserved bytes.
        with tempfile.TemporaryDirectory() as td:
            hoct_src = Path(td) / "hoct_src"
            _write_file(hoct_src / "hoct" / "__init__.py", b"# stub\n")
            _write_file(hoct_src / "hoct" / "create_graph_from_points.py",
                        b"def create_graph_from_points(*a, **k):\n    return {}\n")
            hoct_wts = Path(td) / "weights"
            _write_file(hoct_wts / "general_v0.pth", b"\x01" * 8)
            out = Path(td) / "bundle"
            build(comp_dir=COMP_DIR, bundle_root=out,
                  hoct_source=hoct_src, hoct_weights=hoct_wts,
                  allow_hoct_missing=False, marker_note="")
            prov = json.loads(
                (out / "vendor" / "royerlab_cellmot" / "PROVENANCE.json").read_text()
            )
            for rel, meta in prov["files"].items():
                p = out / "vendor" / "royerlab_cellmot" / rel
                self.assertTrue(p.is_file(), rel)
                self.assertEqual(sha256_file(p), meta["sha256"], rel)


class ManifestVerifierTests(unittest.TestCase):
    def test_missing_marker_file_returns_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m = load_manifest(root, BUNDLE_MANIFEST_FILENAME)
            self.assertEqual(m.get("error"), "manifest_file_missing")

    def test_missing_slot_yields_partial_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_file(root / "vendor" / "x.py", b"y = 1\n")
            manifest = {
                "schema_version": 1,
                "bundle_marker": BUNDLE_MARKER_FILENAME,
                "provenance": {},
                "slots": {
                    SLOT_VENDOR_METRIC: {
                        "name": SLOT_VENDOR_METRIC, "status": STATUS_PRESENT,
                        "files": [{"path": "vendor/x.py",
                                   "sha256": sha256_file(root / "vendor" / "x.py"),
                                   "bytes": 5}],
                        "notes": ""},
                    SLOT_M39_M40:      {"name": SLOT_M39_M40,      "status": STATUS_MISSING, "files": [], "notes": ""},
                    SLOT_M19C:         {"name": SLOT_M19C,         "status": STATUS_MISSING, "files": [], "notes": ""},
                    SLOT_HOCT_SOURCE:  {"name": SLOT_HOCT_SOURCE,  "status": STATUS_MISSING, "files": [], "notes": ""},
                    SLOT_HOCT_WEIGHTS: {"name": SLOT_HOCT_WEIGHTS, "status": STATUS_MISSING, "files": [], "notes": ""},
                },
            }
            v = verify_manifest(root, manifest)
            # metric matches, others missing -> not VERIFIED
            self.assertEqual(v["verdict"], "MISSING_METRIC_OR_ARCHIVE")


if __name__ == "__main__":
    unittest.main()
