"""JSON schemas + validators for M39/M40 artifacts.

Every value that MUST come from the real Kaggle environment carries the
sentinel ``PENDING_REAL_ENV``; the ``validate_*`` helpers permit that
sentinel in every "real-env" field but require it to be REPLACED with a
concrete value before the artifact can be used to decide OK_TO_SUBMIT.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import PENDING_REAL_ENV


# --------------------------------------------------------------------------- #
# CV per-dataset row schema (M39-C: artifacts/m39_cv_results.csv)
# --------------------------------------------------------------------------- #
CV_ROW_COLUMNS: tuple[str, ...] = (
    "dataset",
    "config_id",
    "adjusted_edge_jaccard",
    "division_tp",
    "division_fp",
    "division_fn",
    "division_jaccard",
    "node_count_penalty",
    "n_nodes",
    "n_edges",
    "n_divisions",
    "runtime_seconds",
    "peak_memory_mb",
    "official_score",
    "status",         # "PENDING_REAL_ENV" | "computed" | "failed"
)

# Fields that MUST be concrete numbers before OK_TO_SUBMIT.
CV_ROW_REAL_ENV_FIELDS: tuple[str, ...] = (
    "adjusted_edge_jaccard",
    "division_tp", "division_fp", "division_fn", "division_jaccard",
    "node_count_penalty",
    "n_nodes", "n_edges", "n_divisions",
    "runtime_seconds", "peak_memory_mb",
    "official_score",
)


def make_pending_cv_row(*, dataset: str, config_id: str) -> dict[str, Any]:
    """Return a CV row with only identifier fields set and every real-env
    field marked PENDING_REAL_ENV. Callers MUST replace the sentinels with
    real-environment values before including the row in a submission
    decision."""
    row: dict[str, Any] = {c: PENDING_REAL_ENV for c in CV_ROW_COLUMNS}
    row["dataset"] = dataset
    row["config_id"] = config_id
    row["status"] = PENDING_REAL_ENV
    return row


def validate_cv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return {'ok': bool, 'missing_columns': [...], 'pending_fields': [...],
    'reason': str}. Rows with any PENDING_REAL_ENV in the real-env fields
    are structurally valid but MUST NOT be used to decide OK_TO_SUBMIT."""
    missing = [c for c in CV_ROW_COLUMNS if c not in row]
    pending = [
        c for c in CV_ROW_REAL_ENV_FIELDS
        if row.get(c) == PENDING_REAL_ENV
    ]
    return {
        "ok": not missing,
        "missing_columns": missing,
        "pending_fields": pending,
        "reason": (
            "missing_columns" if missing else
            "pending_real_env_values" if pending else
            "ok"
        ),
    }


# --------------------------------------------------------------------------- #
# CV summary schema (M39-C: artifacts/m39_cv_summary.json)
# --------------------------------------------------------------------------- #
CV_SUMMARY_KEYS: tuple[str, ...] = (
    "metric_name",
    "official_metric_audit_ok",   # bool from metric_vendor audit
    "split_audit_ok",              # bool from split_audit
    "n_configs",
    "n_datasets",
    "macro_official_score",
    "aggregate_official_score",
    "per_config_scores",           # dict[config_id, score]
    "runtime_seconds_total",
    "peak_memory_mb",
    "winner_config_id",
    "winner_official_score",
    "notes",
    "status",                      # "PENDING_REAL_ENV" | "final"
)
CV_SUMMARY_REAL_ENV_FIELDS: tuple[str, ...] = (
    "macro_official_score",
    "aggregate_official_score",
    "runtime_seconds_total",
    "peak_memory_mb",
    "winner_config_id",
    "winner_official_score",
)


def make_pending_cv_summary(*, metric_name: str = PENDING_REAL_ENV) -> dict[str, Any]:
    d = {k: PENDING_REAL_ENV for k in CV_SUMMARY_KEYS}
    d["metric_name"] = metric_name
    d["official_metric_audit_ok"] = False
    d["split_audit_ok"] = False
    d["per_config_scores"] = {}
    d["notes"] = "scaffold; awaits real Kaggle environment"
    d["status"] = PENDING_REAL_ENV
    return d


def validate_cv_summary(d: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in CV_SUMMARY_KEYS if k not in d]
    pending = [k for k in CV_SUMMARY_REAL_ENV_FIELDS if d.get(k) == PENDING_REAL_ENV]
    gate_ok = bool(d.get("official_metric_audit_ok") and d.get("split_audit_ok"))
    return {
        "ok": not missing,
        "missing_keys": missing,
        "pending_fields": pending,
        "gates_pass": gate_ok and not pending,
        "reason": (
            "missing_keys" if missing else
            "audit_gates_not_passed" if not gate_ok else
            "pending_real_env_values" if pending else
            "ok"
        ),
    }


# --------------------------------------------------------------------------- #
# Runtime report schema (M39-D: artifacts/runtime_report.json)
# --------------------------------------------------------------------------- #
RUNTIME_REPORT_TOP_KEYS: tuple[str, ...] = (
    "policy",
    "elapsed_seconds",
    "current_level",
    "active_phase",
    "phases",
    "events",
)

RUNTIME_PHASES_EXPECTED: tuple[str, ...] = (
    "environment_setup",
    "data_discovery",
    "model_loading",
    "detection",
    "candidate_graph_construction",
    "association_inference",
    "ilp_graph_solve",
    "post_processing",
    "validation",
    "csv_write",
    "fallback",
)


def validate_runtime_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Structural check on a report emitted by RuntimeBudget.report()."""
    missing = [k for k in RUNTIME_REPORT_TOP_KEYS if k not in report]
    phases = report.get("phases") or []
    phase_names = [p.get("name") for p in phases if isinstance(p, dict)]
    known_expected = set(RUNTIME_PHASES_EXPECTED)
    unexpected = [n for n in phase_names if n not in known_expected]
    return {
        "ok": not missing,
        "missing_keys": missing,
        "phase_names": phase_names,
        "unexpected_phase_names": unexpected,
        "expected_phases": list(RUNTIME_PHASES_EXPECTED),
    }


# --------------------------------------------------------------------------- #
# Official-metric audit schema (M39-A: artifacts/m39_official_metric_audit.json)
# --------------------------------------------------------------------------- #
METRIC_AUDIT_KEYS: tuple[str, ...] = (
    "source_repo",
    "pinned_commit",
    "pinned_version",
    "vendored_files",              # [{path, sha256, license, bytes}]
    "license_present",
    "internet_disabled_import_ok",
    "reproduces_edge_jaccard",
    "reproduces_division_jaccard",
    "reproduces_node_penalty",
    "match_distance_micrometres",  # must be 7.0
    "voxel_scale_z",               # must be 1.625
    "voxel_scale_y",               # must be 0.40625
    "voxel_scale_x",               # must be 0.40625
    "silent_fallback_to_local_metric_blocked",
    "synthetic_case_matches",      # dict[case_name, bool]
    "status",                      # "PENDING_REAL_ENV" | "verified" | "rejected"
)
METRIC_AUDIT_HARD_GATES: tuple[str, ...] = (
    "license_present",
    "internet_disabled_import_ok",
    "reproduces_edge_jaccard",
    "reproduces_division_jaccard",
    "reproduces_node_penalty",
    "silent_fallback_to_local_metric_blocked",
)


def make_pending_metric_audit() -> dict[str, Any]:
    d = {k: PENDING_REAL_ENV for k in METRIC_AUDIT_KEYS}
    d["vendored_files"] = []
    d["synthetic_case_matches"] = {}
    d["status"] = PENDING_REAL_ENV
    d["match_distance_micrometres"] = 7.0
    d["voxel_scale_z"] = 1.625
    d["voxel_scale_y"] = 0.40625
    d["voxel_scale_x"] = 0.40625
    return d


def validate_metric_audit(d: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in METRIC_AUDIT_KEYS if k not in d]
    pending = [k for k in METRIC_AUDIT_HARD_GATES if d.get(k) == PENDING_REAL_ENV]
    hard_ok = all(d.get(k) is True for k in METRIC_AUDIT_HARD_GATES)
    scales_ok = (
        d.get("match_distance_micrometres") == 7.0
        and d.get("voxel_scale_z") == 1.625
        and d.get("voxel_scale_y") == 0.40625
        and d.get("voxel_scale_x") == 0.40625
    )
    return {
        "ok": not missing,
        "missing_keys": missing,
        "pending_fields": pending,
        "hard_gates_pass": hard_ok,
        "physical_scales_ok": scales_ok,
        "verdict": (
            "PENDING_REAL_ENV" if pending else
            "VERIFIED" if hard_ok and scales_ok else
            "REJECTED"
        ),
    }


# --------------------------------------------------------------------------- #
# Split-audit schema (M39-B: artifacts/m39_split_audit.json)
# --------------------------------------------------------------------------- #
SPLIT_AUDIT_KEYS: tuple[str, ...] = (
    "split_source",                # "manifest" | "checkpoint_metadata" |
                                    #  "explicit_split_file" | "diagnostic_LODO"
    "split_source_path",
    "split_source_sha256",
    "declared_split_name",         # "split_0" if provable, else "diagnostic_*"
    "reconstruction_from_seed",    # MUST be False
    "train_datasets",
    "validation_datasets",
    "train_validation_overlap",    # MUST be empty
    "membership_csv_path",         # "artifacts/m39_split_membership.csv"
    "membership_csv_sha256",
    "status",                      # "PENDING_REAL_ENV" | "verified" | "diagnostic"
)


def validate_split_audit(d: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in SPLIT_AUDIT_KEYS if k not in d]
    reconstructed = bool(d.get("reconstruction_from_seed"))
    overlap = list(d.get("train_validation_overlap") or [])
    train = set(d.get("train_datasets") or [])
    val = set(d.get("validation_datasets") or [])
    intersect = sorted(train & val)
    leakage = bool(overlap) or bool(intersect)
    diagnostic = str(d.get("declared_split_name", "")).startswith("diagnostic_")
    pending = (
        d.get("status") == PENDING_REAL_ENV
        or d.get("split_source") == PENDING_REAL_ENV
        or d.get("declared_split_name") == PENDING_REAL_ENV
        or (not train and not val)
    )
    return {
        "ok": not missing,
        "missing_keys": missing,
        "leakage_detected": leakage,
        "intersect": intersect,
        "reconstructed_from_seed": reconstructed,
        "diagnostic_split_declared": diagnostic,
        "verdict": (
            "REJECT_SPLIT_LEAKAGE_OR_UNPROVEN"
            if leakage or reconstructed else
            "PENDING_REAL_ENV" if pending else
            "OK_DIAGNOSTIC" if diagnostic else
            "OK_PROVEN"
        ),
    }


# --------------------------------------------------------------------------- #
# HOCT shadow report schema (M40-A: artifacts/m40_hoct_shadow_report.json)
# --------------------------------------------------------------------------- #
HOCT_SHADOW_KEYS: tuple[str, ...] = (
    "hoct_source_url",
    "hoct_release_or_commit",
    "hoct_license",
    "hoct_file_hashes",              # {path: sha256}
    "hoct_model_name",               # "general_v0"
    "hoct_model_sha256",
    "adapter_kind",                  # "label_masks" | "point_to_graph"
    "adapter_schema_tests_pass",     # bool
    "create_graph_from_points_status",  # "used_official" | "vendored_adapter" | "stub_detected_blocked"
    "identical_detections_confirmed",   # bool
    "max_delta_t",                   # must be 1
    "n_neighbors",                   # <= 5
    "no_tta",
    "no_tiling",
    "no_long_gap_pass",
    "strict_ilp_time_limit_s",
    "baseline_official_score",       # PENDING_REAL_ENV
    "hoct_official_score",           # PENDING_REAL_ENV
    "official_score_delta",
    "edge_overlap_fraction",
    "division_precision_baseline",
    "division_precision_hoct",
    "division_recall_baseline",
    "division_recall_hoct",
    "baseline_runtime_seconds",
    "hoct_runtime_seconds",
    "status",                        # PENDING_REAL_ENV | shadow_computed
)
HOCT_SHADOW_REAL_ENV_FIELDS: tuple[str, ...] = (
    "hoct_source_url", "hoct_release_or_commit", "hoct_license",
    "hoct_model_sha256", "identical_detections_confirmed",
    "baseline_official_score", "hoct_official_score",
    "official_score_delta", "edge_overlap_fraction",
    "division_precision_baseline", "division_precision_hoct",
    "division_recall_baseline", "division_recall_hoct",
    "baseline_runtime_seconds", "hoct_runtime_seconds",
)


def make_pending_hoct_shadow() -> dict[str, Any]:
    d = {k: PENDING_REAL_ENV for k in HOCT_SHADOW_KEYS}
    d["hoct_file_hashes"] = {}
    d["adapter_kind"] = PENDING_REAL_ENV
    d["adapter_schema_tests_pass"] = False
    d["create_graph_from_points_status"] = PENDING_REAL_ENV
    d["max_delta_t"] = 1
    d["n_neighbors"] = 3
    d["no_tta"] = True
    d["no_tiling"] = True
    d["no_long_gap_pass"] = True
    d["strict_ilp_time_limit_s"] = 60
    d["hoct_model_name"] = "general_v0"
    d["status"] = PENDING_REAL_ENV
    return d


def validate_hoct_shadow(d: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in HOCT_SHADOW_KEYS if k not in d]
    pending = [k for k in HOCT_SHADOW_REAL_ENV_FIELDS if d.get(k) == PENDING_REAL_ENV]
    stub_blocked = d.get("create_graph_from_points_status") != "stub_detected_blocked"
    max_dt_ok = d.get("max_delta_t") == 1
    n_nb_ok = isinstance(d.get("n_neighbors"), int) and d["n_neighbors"] <= 5
    isolation_ok = bool(
        d.get("no_tta") and d.get("no_tiling") and d.get("no_long_gap_pass")
    )
    return {
        "ok": not missing,
        "missing_keys": missing,
        "pending_fields": pending,
        "adapter_stub_blocked_or_absent": stub_blocked,
        "max_delta_t_ok": max_dt_ok,
        "n_neighbors_ok": n_nb_ok,
        "isolation_configured": isolation_ok,
    }


# --------------------------------------------------------------------------- #
# Public: everything visible
# --------------------------------------------------------------------------- #
__all__ = [
    "PENDING_REAL_ENV",
    "CV_ROW_COLUMNS", "CV_ROW_REAL_ENV_FIELDS",
    "make_pending_cv_row", "validate_cv_row",
    "CV_SUMMARY_KEYS", "CV_SUMMARY_REAL_ENV_FIELDS",
    "make_pending_cv_summary", "validate_cv_summary",
    "RUNTIME_REPORT_TOP_KEYS", "RUNTIME_PHASES_EXPECTED",
    "validate_runtime_report",
    "METRIC_AUDIT_KEYS", "METRIC_AUDIT_HARD_GATES",
    "make_pending_metric_audit", "validate_metric_audit",
    "SPLIT_AUDIT_KEYS", "validate_split_audit",
    "HOCT_SHADOW_KEYS", "HOCT_SHADOW_REAL_ENV_FIELDS",
    "make_pending_hoct_shadow", "validate_hoct_shadow",
]
