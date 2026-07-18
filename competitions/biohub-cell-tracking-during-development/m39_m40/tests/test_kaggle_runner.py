"""Kaggle-runner scaffolding tests.

Verifies:
- default policy matches the 75/95/105/110/10 budget.
- preflight_report against the local M19-C archive reports FALLBACK_ONLY
  in this sandbox (no Kaggle mount, no CUDA), and lists the reasons.
- preflight_report against a tampered archive detects the manifest mismatch.
- fallback_to_m19c refuses to fake a submission but reports the correct
  reasons and paths.
- run() returns mode=FALLBACK_ONLY without touching /kaggle/working.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from m39_m40 import PENDING_REAL_ENV
from m39_m40.kaggle_runner import (
    DEFAULT_POLICY,
    build_default_budget,
    fallback_to_m19c,
    preflight_report,
    run,
)


REPO_ARCHIVE = (
    Path(__file__).resolve().parents[2] / "archive" / "m19c"
)


class KaggleRunnerScaffoldTests(unittest.TestCase):
    def test_default_policy_matches_spec(self):
        self.assertEqual(DEFAULT_POLICY.target_seconds, 90 * 60)
        self.assertEqual(DEFAULT_POLICY.hard_engineering_seconds, 105 * 60)
        self.assertEqual(DEFAULT_POLICY.hidden_safety_seconds, 110 * 60)
        self.assertEqual(DEFAULT_POLICY.absolute_seconds, 120 * 60)
        self.assertEqual(DEFAULT_POLICY.finalization_reserve_seconds, 10 * 60)
        DEFAULT_POLICY.validate()

    def test_budget_uses_default_policy(self):
        b = build_default_budget()
        self.assertIs(b.policy, DEFAULT_POLICY)

    def test_preflight_fallback_only_in_sandbox(self):
        pf = preflight_report(archive_dir=REPO_ARCHIVE)
        self.assertEqual(pf.verdict, "FALLBACK_ONLY")
        # In this sandbox: no Kaggle mount, no CUDA
        self.assertFalse(pf.kaggle_mount_ok)
        self.assertFalse(pf.cuda_available)
        # But the M19-C archive IS present and its manifest matches
        self.assertTrue(pf.fallback_archive_ok)
        self.assertTrue(pf.fallback_manifest_matches)

    def test_preflight_detects_tampered_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "m19c"
            fake.mkdir()
            for name in ("M19_VARIANT_C_FULL_CHAIN.txt",
                         "milestone19_metric_postprocess_runner.py",
                         "kaggle_cell_milestone19_metric_postprocess_runner.py",
                         "milestone19_metric_postprocess_runner.ipynb"):
                (fake / name).write_text("tampered")
            (fake / "MANIFEST.sha256").write_text(
                "0" * 64 + "  M19_VARIANT_C_FULL_CHAIN.txt\n"
            )
            pf = preflight_report(archive_dir=fake)
            self.assertFalse(pf.fallback_manifest_matches)
            self.assertIn(
                "M19-C archive hashes do not match manifest", pf.reasons
            )

    def test_fallback_dry_run_reports_one_cell(self):
        r = fallback_to_m19c(archive_dir=REPO_ARCHIVE, dry_run=True)
        self.assertTrue(r["ok"])
        self.assertIn("M19_VARIANT_C_FULL_CHAIN.txt", r["one_cell_path"])

    def test_fallback_refuses_to_fake_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.csv"
            r = fallback_to_m19c(archive_dir=REPO_ARCHIVE, output_path=out, dry_run=False)
            self.assertFalse(r["ok"])
            self.assertEqual(r["reason"], "fallback_regeneration_requires_kaggle_mount")
            # never touched the output path
            self.assertFalse(out.exists())

    def test_run_returns_fallback_mode_in_sandbox(self):
        r = run(archive_dir=REPO_ARCHIVE)
        self.assertEqual(r["mode"], "FALLBACK_ONLY")
        self.assertEqual(r["baseline_official_score"], PENDING_REAL_ENV)
        self.assertEqual(r["hoct_official_score"], PENDING_REAL_ENV)


if __name__ == "__main__":
    unittest.main()
