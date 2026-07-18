"""Unit tests for CV / runtime / metric-audit / split / HOCT schemas."""

from __future__ import annotations

import unittest

from m39_m40 import PENDING_REAL_ENV
from m39_m40.schemas import (
    CV_ROW_COLUMNS,
    CV_SUMMARY_KEYS,
    HOCT_SHADOW_KEYS,
    METRIC_AUDIT_KEYS,
    RUNTIME_PHASES_EXPECTED,
    SPLIT_AUDIT_KEYS,
    make_pending_cv_row,
    make_pending_cv_summary,
    make_pending_hoct_shadow,
    make_pending_metric_audit,
    validate_cv_row,
    validate_cv_summary,
    validate_hoct_shadow,
    validate_metric_audit,
    validate_runtime_report,
    validate_split_audit,
)


class CvSchemaTests(unittest.TestCase):
    def test_pending_row_is_structurally_valid(self):
        r = make_pending_cv_row(dataset="ds", config_id="cfg")
        v = validate_cv_row(r)
        self.assertTrue(v["ok"])
        self.assertEqual(v["reason"], "pending_real_env_values")
        self.assertIn("official_score", v["pending_fields"])

    def test_missing_column_detected(self):
        r = {c: 0 for c in CV_ROW_COLUMNS if c != "official_score"}
        v = validate_cv_row(r)
        self.assertFalse(v["ok"])
        self.assertIn("official_score", v["missing_columns"])

    def test_row_fully_populated(self):
        r = make_pending_cv_row(dataset="d", config_id="c")
        for f in ("adjusted_edge_jaccard", "division_tp", "division_fp", "division_fn",
                  "division_jaccard", "node_count_penalty", "n_nodes", "n_edges",
                  "n_divisions", "runtime_seconds", "peak_memory_mb", "official_score"):
            r[f] = 0
        r["status"] = "computed"
        v = validate_cv_row(r)
        self.assertTrue(v["ok"])
        self.assertEqual(v["pending_fields"], [])
        self.assertEqual(v["reason"], "ok")


class CvSummaryTests(unittest.TestCase):
    def test_pending_summary_blocks_gates(self):
        d = make_pending_cv_summary()
        v = validate_cv_summary(d)
        self.assertTrue(v["ok"])
        self.assertFalse(v["gates_pass"])

    def test_gates_require_both_audits(self):
        d = make_pending_cv_summary()
        d["official_metric_audit_ok"] = True
        d["split_audit_ok"] = False
        self.assertFalse(validate_cv_summary(d)["gates_pass"])
        d["split_audit_ok"] = True
        # still pending real-env values, so gates_pass stays False
        self.assertFalse(validate_cv_summary(d)["gates_pass"])
        for k in ("macro_official_score", "aggregate_official_score",
                  "runtime_seconds_total", "peak_memory_mb",
                  "winner_config_id", "winner_official_score"):
            d[k] = 0.5
        self.assertTrue(validate_cv_summary(d)["gates_pass"])


class RuntimeReportSchemaTests(unittest.TestCase):
    def test_report_with_expected_phases_passes(self):
        report = {
            "policy": {}, "elapsed_seconds": 0, "current_level": "GREEN",
            "active_phase": None,
            "phases": [{"name": n, "start_elapsed_seconds": 0,
                        "end_elapsed_seconds": 0, "status": "completed",
                        "metadata": {}, "duration_seconds": 0}
                       for n in RUNTIME_PHASES_EXPECTED],
            "events": [],
        }
        v = validate_runtime_report(report)
        self.assertTrue(v["ok"])
        self.assertEqual(v["unexpected_phase_names"], [])

    def test_report_missing_key_detected(self):
        v = validate_runtime_report({"policy": {}})
        self.assertFalse(v["ok"])
        self.assertIn("elapsed_seconds", v["missing_keys"])


class MetricAuditTests(unittest.TestCase):
    def test_pending_audit_verdict(self):
        d = make_pending_metric_audit()
        v = validate_metric_audit(d)
        self.assertEqual(v["verdict"], "PENDING_REAL_ENV")
        self.assertTrue(v["physical_scales_ok"])

    def test_verified_when_all_gates_pass(self):
        d = make_pending_metric_audit()
        for k in ("license_present", "internet_disabled_import_ok",
                  "reproduces_edge_jaccard", "reproduces_division_jaccard",
                  "reproduces_node_penalty",
                  "silent_fallback_to_local_metric_blocked"):
            d[k] = True
        v = validate_metric_audit(d)
        self.assertEqual(v["verdict"], "VERIFIED")

    def test_rejected_on_wrong_scale(self):
        d = make_pending_metric_audit()
        for k in ("license_present", "internet_disabled_import_ok",
                  "reproduces_edge_jaccard", "reproduces_division_jaccard",
                  "reproduces_node_penalty",
                  "silent_fallback_to_local_metric_blocked"):
            d[k] = True
        d["match_distance_micrometres"] = 5.0
        v = validate_metric_audit(d)
        self.assertEqual(v["verdict"], "REJECTED")


class SplitAuditTests(unittest.TestCase):
    def _base(self) -> dict:
        return {k: PENDING_REAL_ENV for k in SPLIT_AUDIT_KEYS} | {
            "reconstruction_from_seed": False,
            "train_datasets": [], "validation_datasets": [],
            "train_validation_overlap": [],
        }

    def test_reject_when_reconstructed_from_seed(self):
        d = self._base(); d["reconstruction_from_seed"] = True
        self.assertEqual(validate_split_audit(d)["verdict"],
                         "REJECT_SPLIT_LEAKAGE_OR_UNPROVEN")

    def test_reject_when_overlap(self):
        d = self._base(); d["train_datasets"] = ["a"]; d["validation_datasets"] = ["a"]
        v = validate_split_audit(d)
        self.assertTrue(v["leakage_detected"])
        self.assertEqual(v["verdict"], "REJECT_SPLIT_LEAKAGE_OR_UNPROVEN")

    def _fill_ok(self, d: dict) -> dict:
        d["status"] = "verified"
        d["split_source"] = "manifest"
        d["split_source_path"] = "reference/kaggle_test_splits_50ep.json"
        d["split_source_sha256"] = "0" * 64
        d["membership_csv_path"] = "artifacts/m39_split_membership.csv"
        d["membership_csv_sha256"] = "0" * 64
        return d

    def test_ok_proven(self):
        d = self._base(); d["train_datasets"] = ["a"]; d["validation_datasets"] = ["b"]
        d["declared_split_name"] = "split_0"; self._fill_ok(d)
        self.assertEqual(validate_split_audit(d)["verdict"], "OK_PROVEN")

    def test_ok_diagnostic(self):
        d = self._base(); d["train_datasets"] = ["a"]; d["validation_datasets"] = ["b"]
        d["declared_split_name"] = "diagnostic_LODO_x"; self._fill_ok(d)
        self.assertEqual(validate_split_audit(d)["verdict"], "OK_DIAGNOSTIC")

    def test_pending_when_all_sentinels(self):
        d = self._base()   # every real-env field is PENDING_REAL_ENV, no train/val
        self.assertEqual(validate_split_audit(d)["verdict"], "PENDING_REAL_ENV")


class HoctShadowTests(unittest.TestCase):
    def test_pending_shadow_report(self):
        d = make_pending_hoct_shadow()
        v = validate_hoct_shadow(d)
        self.assertTrue(v["ok"])
        self.assertTrue(v["max_delta_t_ok"])
        self.assertTrue(v["n_neighbors_ok"])
        self.assertTrue(v["isolation_configured"])

    def test_max_delta_t_gate(self):
        d = make_pending_hoct_shadow(); d["max_delta_t"] = 2
        self.assertFalse(validate_hoct_shadow(d)["max_delta_t_ok"])

    def test_n_neighbors_cap(self):
        d = make_pending_hoct_shadow(); d["n_neighbors"] = 12
        self.assertFalse(validate_hoct_shadow(d)["n_neighbors_ok"])


if __name__ == "__main__":
    unittest.main()
