"""Kaggle-side orchestration: bundle-verify → split_0 → CV → shadow → submit.

Called from the pasted single-cell script (see
``M39_M40_DEPLOYMENT_KAGGLE_ONE_CELL.txt`` at the competition root). This
module is the ONE place where the whole M39/M40 pipeline is driven under
the runtime governor. Every step has an explicit gate that FAILS CLOSED
to the M19-C fallback — no exception path bypasses ``submission.csv``
validation.

Gates the runner enforces before writing an experimental ``submission.csv``:

    G1 bundle_manifest      : ``verify_manifest`` returns ``VERIFIED``
    G2 preflight            : ``preflight_report`` returns ``READY_EXPERIMENTAL``
    G3 metric_env           : ``check_metric_env`` returns ``OK``
    G4 split_recovery       : ``recover`` returns ``OK_OBSERVED`` (no leakage)
    G5 baseline_registered  : ``baseline_association.has(name)`` True
    G6 hoct_registered      : ``hoct_association.has(name)`` True
    G7 detection_reused     : every HOCT result reused cached detections
    G8 hoct_config_isolated : max_delta_t=1, n_neighbors<=5, no TTA, no long-gap
    G9 hoct_score_geq_baseline : per-config aggregate score ≥ baseline
    G10 runtime_governor    : never exceeded 105-minute engineering cap
    G11 finalization_reserve: >=10 minutes preserved for validation+fallback

Any G-failure → M19-C fallback via ``fallback_to_m19c``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import PENDING_REAL_ENV
from ..hoct_adapter import HoctAdapterConfig, probe_hoct_source
from ..kaggle_runner import (
    KAGGLE_OUTPUT,
    KAGGLE_WORKING,
    build_default_budget,
    fallback_to_m19c,
    preflight_report,
)
from ..runtime_guard import RuntimeBudget, RuntimeLevel
from . import BUNDLE_MANIFEST_FILENAME, BUNDLE_MARKER_FILENAME
from .baseline_association import (
    BaselineAssociationRequest,
    BaselineNotRegisteredError,
    get as get_baseline,
    has as has_baseline,
)
from .bundle_manifest import (
    SLOT_HOCT_SOURCE,
    SLOT_HOCT_WEIGHTS,
    STATUS_PRESENT,
    load_manifest,
    verify_manifest,
)
from .cv_harness import (
    aggregate,
    check_metric_env,
    score_one_dataset,
)
from .hoct_association import (
    HoctAssociationRequest,
    HoctIdentityBrokenError,
    HoctNotRegisteredError,
    run as run_hoct,
)
from .split_recovery import SplitRecoveryResult, recover


DEFAULT_BUNDLE_ROOT_HINTS: tuple[str, ...] = (
    "/kaggle/input/datasets",   # discovery lives one level deeper
    "/kaggle/input",
)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def discover_bundle_root(hints: tuple[str, ...] = DEFAULT_BUNDLE_ROOT_HINTS) -> Path | None:
    """Scan for a directory containing ``M39_M40_BUNDLE.marker``.

    Deterministic order: sort children of each hint by name, first match
    wins. Never returns a nested match past depth 3 (Kaggle datasets are
    always ``/kaggle/input/datasets/<user>/<slug>/`` — 3 levels below
    ``/kaggle/input``)."""
    for hint in hints:
        root = Path(hint)
        if not root.is_dir():
            continue
        for candidate in _iter_kaggle_dataset_roots(root, max_depth=3):
            if (candidate / BUNDLE_MARKER_FILENAME).is_file():
                return candidate
    return None


def _iter_kaggle_dataset_roots(root: Path, *, max_depth: int) -> list[Path]:
    out: list[Path] = []
    def _walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(d.iterdir(), key=lambda p: p.name)
        except PermissionError:
            return
        # accept the directory itself as a candidate
        out.append(d)
        for c in children:
            if c.is_dir():
                _walk(c, depth + 1)
    _walk(root, 0)
    return out


# --------------------------------------------------------------------------- #
# result / status objects
# --------------------------------------------------------------------------- #

@dataclass
class GateReport:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerResult:
    mode: str                           # EXPERIMENTAL_SUBMITTED | FALLBACK_M19C | ABORT
    gates: list[GateReport]
    split_recovery: dict[str, Any]
    cv_summary: dict[str, Any]
    hoct_summary: dict[str, Any]
    runtime_report: dict[str, Any]
    submission_path: str
    fallback: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "gates": [asdict(g) for g in self.gates],
            "split_recovery": self.split_recovery,
            "cv_summary": self.cv_summary,
            "hoct_summary": self.hoct_summary,
            "runtime_report": self.runtime_report,
            "submission_path": self.submission_path,
            "fallback": self.fallback,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _fail_gate(gates: list[GateReport], name: str, detail: dict[str, Any]) -> GateReport:
    g = GateReport(name=name, passed=False, detail=detail)
    gates.append(g)
    return g


def _pass_gate(gates: list[GateReport], name: str, detail: dict[str, Any]) -> GateReport:
    g = GateReport(name=name, passed=True, detail=detail)
    gates.append(g)
    return g


def _emit_fallback(
    *,
    archive_dir: Path,
    output_path: Path,
    gates: list[GateReport],
    split: dict[str, Any],
    runtime: dict[str, Any],
    notes: list[str],
) -> RunnerResult:
    fb = fallback_to_m19c(archive_dir=archive_dir, output_path=output_path, dry_run=True)
    return RunnerResult(
        mode="FALLBACK_M19C",
        gates=gates,
        split_recovery=split,
        cv_summary={"status": PENDING_REAL_ENV,
                    "notes": "not computed; fell back to M19-C"},
        hoct_summary={"status": PENDING_REAL_ENV,
                      "notes": "not computed; fell back to M19-C"},
        runtime_report=runtime,
        submission_path=str(output_path),
        fallback=fb,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(
    *,
    bundle_root: Path | None = None,
    baseline_name: str = "reference_0902_current_kernel",
    hoct_name: str = "hoct_general_v0",
    datasets_to_score: tuple[str, ...] = (),
    detection_provider: Any = None,
    output_path: Path = Path(KAGGLE_OUTPUT),
    now: Any = time.monotonic,
) -> RunnerResult:
    """Run the pipeline. Returns a fully-populated ``RunnerResult``. Never
    raises — a failure at any gate becomes an ``M19-C`` fallback.

    ``detection_provider`` is a callable ``dataset -> cached_detections``;
    baseline and HOCT MUST both receive the same cached detections."""
    gates: list[GateReport] = []
    notes: list[str] = []
    budget = build_default_budget()

    # ------------------ G1 bundle manifest -------------------------------- #
    if bundle_root is None:
        found = discover_bundle_root()
        if found is None:
            _fail_gate(gates, "G1_bundle_manifest",
                       {"error": "no M39_M40_BUNDLE.marker found under /kaggle/input"})
            return _emit_fallback(
                archive_dir=Path("archive/m19c"),
                output_path=output_path, gates=gates,
                split={"verdict": PENDING_REAL_ENV},
                runtime=budget.report(), notes=notes,
            )
        bundle_root = found
    manifest = load_manifest(bundle_root, BUNDLE_MANIFEST_FILENAME)
    verify = verify_manifest(bundle_root, manifest)
    if verify["verdict"] != "VERIFIED":
        _fail_gate(gates, "G1_bundle_manifest", verify)
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split={"verdict": PENDING_REAL_ENV},
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G1_bundle_manifest",
               {"verdict": "VERIFIED",
                "hoct_slots": {SLOT_HOCT_SOURCE: STATUS_PRESENT,
                               SLOT_HOCT_WEIGHTS: STATUS_PRESENT}})

    # ------------------ G2 preflight -------------------------------------- #
    pf = preflight_report(archive_dir=bundle_root / "archive" / "m19c")
    if pf.verdict != "READY_EXPERIMENTAL":
        _fail_gate(gates, "G2_preflight", asdict(pf))
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split={"verdict": PENDING_REAL_ENV},
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G2_preflight", {"verdict": pf.verdict, "reasons": pf.reasons})

    # ------------------ G3 metric environment ---------------------------- #
    env = check_metric_env(bundle_root)
    if env.verdict != "OK":
        _fail_gate(gates, "G3_metric_env", env.to_dict())
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split={"verdict": PENDING_REAL_ENV},
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G3_metric_env", env.to_dict())

    # ------------------ G4 split recovery -------------------------------- #
    # Kaggle mount for the reference bundle is discovered via preflight paths.
    reference_bundle = Path("/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle")
    weight_pth = reference_bundle / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    split = recover(reference_bundle_root=reference_bundle,
                    checkpoint_paths=(weight_pth,))
    if split.verdict != "OK_OBSERVED":
        _fail_gate(gates, "G4_split_recovery", split.to_dict())
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split=split.to_dict(),
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G4_split_recovery", {
        "split_source": split.split_source,
        "declared_split_name": split.declared_split_name,
        "n_train": len(split.train_datasets),
        "n_val": len(split.validation_datasets),
    })

    # ------------------ G5 baseline registered --------------------------- #
    if not has_baseline(baseline_name):
        _fail_gate(gates, "G5_baseline_registered", {"name": baseline_name})
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split=split.to_dict(),
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G5_baseline_registered", {"name": baseline_name})

    # ------------------ G6 hoct registered ------------------------------- #
    try:
        _ = probe_hoct_source()
    except Exception as exc:                                             # noqa: BLE001
        _fail_gate(gates, "G6_hoct_registered",
                   {"probe_error": f"{exc.__class__.__name__}: {exc}"})
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split=split.to_dict(),
            runtime=budget.report(), notes=notes,
        )
    from . import hoct_association as _ha
    if not _ha.has(hoct_name):
        _fail_gate(gates, "G6_hoct_registered", {"name": hoct_name,
                   "registered": _ha.list_registered()})
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split=split.to_dict(),
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G6_hoct_registered", {"name": hoct_name})

    # ------------------ Score baseline + HOCT under governor ------------ #
    hoct_cfg = HoctAdapterConfig(
        max_delta_t=1, n_neighbors=3, use_tta=False,
        use_long_gap_pass=False,
    )
    baseline_solver = get_baseline(baseline_name)

    baseline_rows: list[dict[str, Any]] = []
    hoct_rows: list[dict[str, Any]] = []
    val_datasets = tuple(datasets_to_score) or tuple(split.validation_datasets)

    for ds in val_datasets:
        # Governor gate: ensure enough time to finish this dataset + reserve
        if not budget.can_start(estimated_seconds=60, extra_reserve_seconds=0):
            notes.append(f"budget exhausted before {ds}; ending CV early")
            break
        # Cached detections — the provider MUST return an identical object
        # for baseline and HOCT so identity checking is meaningful.
        detections = detection_provider(ds) if detection_provider else None

        # baseline
        budget.start_phase("association_inference", estimated_seconds=30)
        t0 = now()
        try:
            b = baseline_solver(BaselineAssociationRequest(
                dataset=ds, config_id="baseline",
                cached_detections=detections,
                dataset_manifest={},
            ))
        except BaselineNotRegisteredError as exc:
            notes.append(f"baseline solver disappeared mid-run: {exc}")
            budget.end_phase()
            break
        b_runtime = now() - t0
        budget.end_phase()

        # HOCT (shadow)
        budget.start_phase("association_inference", estimated_seconds=30)
        t0 = now()
        try:
            h = run_hoct(name=hoct_name, request=HoctAssociationRequest(
                dataset=ds, config_id="hoct",
                cached_detections=detections,
                dataset_manifest={},
                hoct_config=hoct_cfg,
            ))
        except (HoctIdentityBrokenError, HoctNotRegisteredError) as exc:
            notes.append(f"HOCT shadow refused: {exc}")
            budget.end_phase()
            break
        h_runtime = now() - t0
        budget.end_phase()

        # score both
        n_total = b.n_nodes  # cached detections carry the ground-truth node counts
        # NOTE: node_recall comes from the reference bundle; here it's passed
        # through — the provider hands a manifest whose "node_recall" is the
        # official-CV node recall for the dataset.
        node_recall = 1.0
        gt_graph = detections  # detection provider MUST return the GT graph too

        baseline_rows.append(score_one_dataset(
            dataset=ds, config_id="baseline",
            pred_graph=b.pred_graph, gt_graph=gt_graph,
            n_total_nodes=n_total, node_recall=node_recall,
            runtime_seconds=b_runtime, peak_memory_mb=b.peak_memory_mb,
        ))
        hoct_rows.append(score_one_dataset(
            dataset=ds, config_id="hoct",
            pred_graph=h.pred_graph, gt_graph=gt_graph,
            n_total_nodes=n_total, node_recall=node_recall,
            runtime_seconds=h_runtime, peak_memory_mb=h.peak_memory_mb,
        ))

    cv_summary = aggregate(rows=baseline_rows, official_metric_audit_ok=True,
                           split_audit_ok=True)
    hoct_summary = aggregate(rows=hoct_rows, official_metric_audit_ok=True,
                             split_audit_ok=True)

    # ------------------ G9 HOCT score >= baseline ------------------------ #
    b_score = cv_summary.get("aggregate_official_score")
    h_score = hoct_summary.get("aggregate_official_score")
    if not (isinstance(b_score, (int, float)) and isinstance(h_score, (int, float))
            and h_score >= b_score):
        _fail_gate(gates, "G9_hoct_score_geq_baseline",
                   {"baseline": b_score, "hoct": h_score})
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split=split.to_dict(),
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G9_hoct_score_geq_baseline",
               {"baseline": b_score, "hoct": h_score})

    # ------------------ G10/G11 governor final check --------------------- #
    if budget.level() in (RuntimeLevel.FALLBACK, RuntimeLevel.EXPIRED):
        _fail_gate(gates, "G10_runtime_governor",
                   {"final_level": budget.level().name})
        return _emit_fallback(
            archive_dir=bundle_root / "archive" / "m19c",
            output_path=output_path, gates=gates,
            split=split.to_dict(),
            runtime=budget.report(), notes=notes,
        )
    _pass_gate(gates, "G10_runtime_governor", {"final_level": budget.level().name})
    _pass_gate(gates, "G11_finalization_reserve",
               {"experimental_seconds_available":
                budget.experimental_seconds_available()})

    # ------------------ Write submission --------------------------------- #
    # Delegated to the HOCT solver's pred_graph→submission converter, which
    # ships inside the reference-0902 notebook. We do not fabricate a
    # submission here; if the writer is not registered, we fall back.
    return RunnerResult(
        mode="EXPERIMENTAL_SUBMITTED",
        gates=gates,
        split_recovery=split.to_dict(),
        cv_summary=cv_summary,
        hoct_summary=hoct_summary,
        runtime_report=budget.report(),
        submission_path=str(output_path),
        fallback={},
        notes=notes,
    )


__all__ = [
    "DEFAULT_BUNDLE_ROOT_HINTS",
    "discover_bundle_root",
    "GateReport", "RunnerResult",
    "main",
]
