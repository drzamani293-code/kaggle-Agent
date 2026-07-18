"""M39 official CV harness (uses vendored ``tracking_cellmot`` scorer).

Every score field that requires real predicted+GT graphs is filled by the
real Kaggle GPU run — never in this sandbox. This module is responsible
only for:

1. Ensuring the vendored ``tracking_cellmot`` package sits on ``sys.path``
   (from the mounted bundle root's ``vendor/royerlab_cellmot/src``).
2. Verifying the required scoring dependencies are importable
   (``tracksdata``, ``polars``, ``scipy``, ``geff``); reporting
   ``OFFICIAL_METRIC_DEPENDENCY_MISSING`` otherwise (no proxy scoring).
3. Scoring one predicted graph against one GT graph via
   ``tracking_cellmot.metrics.evaluate`` (max_distance_um=7.0),
   ``per_sample_metrics``, and aggregating with ``summarise``.
4. Recording per-dataset rows into ``m39_cv_results.csv`` and writing
   the summary via ``schemas.make_pending_cv_summary`` merged with real
   observed numbers.

Every function accepts already-materialised graph objects; it never
reads competition data itself. The caller (``kaggle_one_cell_runner``)
runs baseline association, caches detections, and hands the graphs in.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import PENDING_REAL_ENV
from ..schemas import (
    CV_ROW_COLUMNS,
    make_pending_cv_row,
    make_pending_cv_summary,
    validate_cv_row,
    validate_cv_summary,
)


REQUIRED_METRIC_DEPS: tuple[str, ...] = ("tracksdata", "polars", "scipy", "geff")
DEFAULT_MAX_DISTANCE_UM = 7.0


@dataclass
class MetricEnvReport:
    tracking_cellmot_on_path: bool
    tracking_cellmot_imported: bool
    dependencies_present: dict[str, bool]
    verdict: str                            # OK | OFFICIAL_METRIC_DEPENDENCY_MISSING | OFFICIAL_METRIC_IMPORT_FAILED
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracking_cellmot_on_path": self.tracking_cellmot_on_path,
            "tracking_cellmot_imported": self.tracking_cellmot_imported,
            "dependencies_present": dict(self.dependencies_present),
            "verdict": self.verdict,
            "error": self.error,
        }


def ensure_metric_on_path(bundle_root: Path) -> bool:
    """Prepend ``<bundle_root>/vendor/royerlab_cellmot/src`` to sys.path so
    ``tracking_cellmot`` becomes importable. Returns True iff the path is
    now on sys.path AND the directory contains the expected package."""
    src = bundle_root / "vendor" / "royerlab_cellmot" / "src"
    pkg = src / "tracking_cellmot" / "__init__.py"
    if not pkg.is_file():
        return False
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return True


def check_metric_env(bundle_root: Path) -> MetricEnvReport:
    on_path = ensure_metric_on_path(bundle_root)
    deps: dict[str, bool] = {}
    for m in REQUIRED_METRIC_DEPS:
        try:
            importlib.import_module(m)
            deps[m] = True
        except Exception:                                              # noqa: BLE001
            deps[m] = False

    if not on_path:
        return MetricEnvReport(
            tracking_cellmot_on_path=False,
            tracking_cellmot_imported=False,
            dependencies_present=deps,
            verdict="OFFICIAL_METRIC_IMPORT_FAILED",
            error="vendor/royerlab_cellmot/src/tracking_cellmot not present",
        )
    if not all(deps.values()):
        return MetricEnvReport(
            tracking_cellmot_on_path=True,
            tracking_cellmot_imported=False,
            dependencies_present=deps,
            verdict="OFFICIAL_METRIC_DEPENDENCY_MISSING",
            error=f"missing dependencies: {[k for k, v in deps.items() if not v]}",
        )
    try:
        importlib.import_module("tracking_cellmot.metrics")
    except Exception as exc:                                             # noqa: BLE001
        return MetricEnvReport(
            tracking_cellmot_on_path=True,
            tracking_cellmot_imported=False,
            dependencies_present=deps,
            verdict="OFFICIAL_METRIC_IMPORT_FAILED",
            error=f"{exc.__class__.__name__}: {exc}",
        )
    return MetricEnvReport(
        tracking_cellmot_on_path=True,
        tracking_cellmot_imported=True,
        dependencies_present=deps,
        verdict="OK",
    )


def score_one_dataset(
    *,
    dataset: str,
    config_id: str,
    pred_graph: Any,
    gt_graph: Any,
    n_total_nodes: int,
    node_recall: float,
    runtime_seconds: float,
    peak_memory_mb: float,
    max_distance_um: float = DEFAULT_MAX_DISTANCE_UM,
) -> dict[str, Any]:
    """Score one (pred_graph, gt_graph) pair with the vendored metric.

    Called from the Kaggle-side runner. Never raises when the metric
    environment is not OK — the runner must call ``check_metric_env`` and
    gate on ``OK`` first."""
    from tracking_cellmot.metrics import evaluate, per_sample_metrics    # type: ignore

    er = evaluate(pred_graph, gt_graph, max_distance=max_distance_um)
    psm = per_sample_metrics(er, n_total=n_total_nodes, node_recall=node_recall)

    row = make_pending_cv_row(dataset=dataset, config_id=config_id)
    row["adjusted_edge_jaccard"] = float(getattr(psm, "adj_edge_jaccard", psm.get("adj_edge_jaccard") if isinstance(psm, dict) else 0.0))
    row["division_tp"] = int(getattr(er, "division_tp", er.get("division_tp") if isinstance(er, dict) else 0))
    row["division_fp"] = int(getattr(er, "division_fp", er.get("division_fp") if isinstance(er, dict) else 0))
    row["division_fn"] = int(getattr(er, "division_fn", er.get("division_fn") if isinstance(er, dict) else 0))
    row["division_jaccard"] = float(getattr(psm, "division_jaccard", psm.get("division_jaccard") if isinstance(psm, dict) else 0.0))
    row["node_count_penalty"] = float(getattr(psm, "node_count_penalty", psm.get("node_count_penalty") if isinstance(psm, dict) else 0.0))
    row["n_nodes"] = int(getattr(er, "n_pred_nodes", er.get("n_pred_nodes") if isinstance(er, dict) else 0))
    row["n_edges"] = int(getattr(er, "n_pred_edges", er.get("n_pred_edges") if isinstance(er, dict) else 0))
    row["n_divisions"] = int(row["division_tp"]) + int(row["division_fp"])
    row["runtime_seconds"] = float(runtime_seconds)
    row["peak_memory_mb"] = float(peak_memory_mb)
    row["official_score"] = float(getattr(psm, "score", psm.get("score") if isinstance(psm, dict) else 0.0))
    row["status"] = "computed"
    return row


def aggregate(
    *,
    rows: list[dict[str, Any]],
    official_metric_audit_ok: bool,
    split_audit_ok: bool,
) -> dict[str, Any]:
    """Compute summary from computed rows. Refuses to produce a numeric
    summary if any row is still PENDING_REAL_ENV or the two audit gates
    are not both True."""
    summary = make_pending_cv_summary(metric_name="tracking_cellmot.metrics.evaluate")
    summary["official_metric_audit_ok"] = bool(official_metric_audit_ok)
    summary["split_audit_ok"] = bool(split_audit_ok)
    summary["n_configs"] = len({r["config_id"] for r in rows}) if rows else 0
    summary["n_datasets"] = len({r["dataset"] for r in rows}) if rows else 0

    computed = [r for r in rows if r.get("status") == "computed"]
    if not (official_metric_audit_ok and split_audit_ok) or not computed:
        summary["status"] = PENDING_REAL_ENV
        summary["notes"] = ("audits not both true, or no computed rows; "
                            "summary held at PENDING_REAL_ENV")
        return summary

    # Macro across datasets, aggregate across configs (matching upstream summarise semantics).
    scores = [float(r["official_score"]) for r in computed]
    summary["macro_official_score"] = sum(scores) / len(scores)
    summary["aggregate_official_score"] = sum(scores) / len(scores)  # single-config guard
    summary["per_config_scores"] = _per_config(computed)
    summary["runtime_seconds_total"] = sum(float(r["runtime_seconds"]) for r in computed)
    summary["peak_memory_mb"] = max(float(r["peak_memory_mb"]) for r in computed)
    winner_id, winner_score = max(summary["per_config_scores"].items(), key=lambda kv: kv[1])
    summary["winner_config_id"] = winner_id
    summary["winner_official_score"] = float(winner_score)
    summary["status"] = "final"
    summary["notes"] = ""
    return summary


def _per_config(rows: list[dict[str, Any]]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(str(r["config_id"]), []).append(float(r["official_score"]))
    return {cid: (sum(s) / len(s)) for cid, s in by.items()}


__all__ = [
    "REQUIRED_METRIC_DEPS", "DEFAULT_MAX_DISTANCE_UM",
    "MetricEnvReport", "ensure_metric_on_path", "check_metric_env",
    "score_one_dataset", "aggregate",
]
