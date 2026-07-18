"""Kaggle real-environment runner + checklist (M39-D / M40-A).

This module is the ONE place where the real Kaggle GPU environment is
allowed to invoke:

- vendored official metric,
- baseline detector + baseline association,
- HOCT shadow adapter (identical detections),
- runtime governor with the 105-min engineering cap, 110-min hidden safety
  ceiling, and 10-min finalization reserve.

It never runs in this sandbox — every phase is gated on the
``check_prereqs`` result. Instead, this module exposes:

1. ``preflight_report`` — a static checklist you can print to any Kaggle
   cell to verify the environment is ready.
2. ``build_default_budget`` — a preconfigured ``RuntimeBudget`` with the
   105/110/10 policy.
3. ``fallback_to_m19c`` — emit the byte-preserved M19-C submission when the
   experimental path cannot finish safely; guarded so it never
   accidentally deletes a valid non-fallback submission.
4. ``run`` — the orchestration entry point invoked only when the checklist
   is fully green.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import PENDING_REAL_ENV
from .hoct_adapter import HoctAdapterConfig, check_prereqs, probe_hoct_source
from .runtime_guard import RuntimeBudget, RuntimeLevel, RuntimePolicy


# ------------------------------ config ------------------------------------ #

DEFAULT_POLICY = RuntimePolicy(
    target_seconds=90 * 60,
    hard_engineering_seconds=105 * 60,
    hidden_safety_seconds=110 * 60,
    absolute_seconds=120 * 60,
    finalization_reserve_seconds=10 * 60,
)

KAGGLE_COMP_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
KAGGLE_REFERENCE_BUNDLE = "/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle"
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_OUTPUT = "/kaggle/working/submission.csv"

# Paths that must exist in the real Kaggle environment for the experimental
# path to proceed; missing paths force fallback to M19-C.
REQUIRED_KAGGLE_INPUTS: tuple[str, ...] = (
    KAGGLE_COMP_DIR,
    KAGGLE_COMP_DIR + "/test",
    KAGGLE_REFERENCE_BUNDLE,
    KAGGLE_REFERENCE_BUNDLE + "/reference/biohub-competition-solution.ipynb",
    KAGGLE_REFERENCE_BUNDLE + "/weights/unet_transformer/split_0/edge_predictor_best.pth",
)


# ------------------------------ preflight --------------------------------- #

@dataclass
class PreflightReport:
    kaggle_mount_ok: bool
    required_inputs: list[dict[str, Any]]
    torch_importable: bool
    cuda_available: bool
    hoct_probe: dict[str, Any]
    fallback_archive_ok: bool
    fallback_manifest_matches: bool
    verdict: str                       # "READY_EXPERIMENTAL" | "FALLBACK_ONLY"
    reasons: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_check(archive_dir: Path) -> tuple[bool, bool]:
    manifest = archive_dir / "MANIFEST.sha256"
    if not manifest.exists():
        return False, False
    try:
        want: dict[str, str] = {}
        for line in manifest.read_text().splitlines():
            sha, name = line.split("  ", 1)
            want[name] = sha
        all_ok = True
        for name, sha in want.items():
            p = archive_dir / name
            if not p.is_file() or _sha256(p) != sha:
                all_ok = False
                break
        return True, all_ok
    except (OSError, ValueError):
        return True, False


def preflight_report(
    *,
    archive_dir: Path,
    required_inputs: tuple[str, ...] = REQUIRED_KAGGLE_INPUTS,
) -> PreflightReport:
    inputs = [
        {"path": p, "exists": Path(p).exists()}
        for p in required_inputs
    ]
    kaggle_mount_ok = all(i["exists"] for i in inputs)

    torch_ok = False
    cuda_ok = False
    try:
        import torch  # type: ignore
        torch_ok = True
        cuda_ok = bool(torch.cuda.is_available())          # type: ignore[attr-defined]
    except Exception:                                        # noqa: BLE001
        pass

    hoct_probe = probe_hoct_source()
    archive_present, archive_ok = _archive_check(archive_dir)

    reasons: list[str] = []
    if not kaggle_mount_ok:
        reasons.append("required Kaggle inputs missing")
    if not torch_ok:
        reasons.append("torch not importable")
    if not cuda_ok:
        reasons.append("no CUDA GPU available")
    if hoct_probe.get("create_graph_from_points_status") == "stub_detected_blocked":
        reasons.append("HOCT create_graph_from_points is a stub")
    if not archive_present:
        reasons.append("M19-C archive MANIFEST.sha256 missing")
    if archive_present and not archive_ok:
        reasons.append("M19-C archive hashes do not match manifest")

    experimental_ok = (
        kaggle_mount_ok and torch_ok and cuda_ok
        and hoct_probe.get("create_graph_from_points_status") == "used_official"
        and archive_present and archive_ok
    )
    return PreflightReport(
        kaggle_mount_ok=kaggle_mount_ok,
        required_inputs=inputs,
        torch_importable=torch_ok,
        cuda_available=cuda_ok,
        hoct_probe=hoct_probe,
        fallback_archive_ok=archive_present and archive_ok,
        fallback_manifest_matches=archive_ok,
        verdict="READY_EXPERIMENTAL" if experimental_ok else "FALLBACK_ONLY",
        reasons=reasons,
    )


# ------------------------------ budget ------------------------------------ #

def build_default_budget() -> RuntimeBudget:
    return RuntimeBudget(policy=DEFAULT_POLICY)


# ------------------------------ fallback ---------------------------------- #

def fallback_to_m19c(
    *,
    archive_dir: Path,
    output_path: Path = Path(KAGGLE_OUTPUT),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Emit an M19-C-equivalent submission by copying the one-cell script's
    generated output when possible; when only the source runner is available
    in the archive, the caller must execute the runner in the Kaggle
    environment (this scaffold refuses to fake it).

    Returns a status dict. Never raises; never overwrites a non-fallback
    submission unless ``dry_run=False`` and the caller has explicitly
    decided to fall back."""
    manifest = archive_dir / "MANIFEST.sha256"
    if not manifest.exists():
        return {"ok": False, "reason": "manifest_missing", "output_path": str(output_path)}
    one_cell = archive_dir / "M19_VARIANT_C_FULL_CHAIN.txt"
    if not one_cell.exists():
        return {"ok": False, "reason": "one_cell_missing", "output_path": str(output_path)}
    if dry_run:
        return {"ok": True, "reason": "dry_run", "one_cell_path": str(one_cell),
                "note": "in Kaggle: exec the one-cell file to regenerate submission.csv"}
    # In the real Kaggle environment M19-C regenerates submission.csv by
    # exec'ing its own one-cell script; this scaffold cannot do that without
    # the mounted reference bundle. Refuse rather than fake:
    return {
        "ok": False,
        "reason": "fallback_regeneration_requires_kaggle_mount",
        "one_cell_path": str(one_cell),
        "output_path": str(output_path),
    }


# ------------------------------ orchestration ----------------------------- #

def run(
    *,
    archive_dir: Path,
    output_path: Path = Path(KAGGLE_OUTPUT),
    hoct_config: HoctAdapterConfig | None = None,
) -> dict[str, Any]:
    """Full experimental orchestration. Only proceeds when the preflight
    reports READY_EXPERIMENTAL; otherwise emits M19-C fallback.

    In this sandbox the preflight will always be FALLBACK_ONLY; the run
    returns without touching the working directory. All CV scores and
    runtimes carry ``PENDING_REAL_ENV`` in the report."""
    hoct_config = hoct_config or HoctAdapterConfig()
    hoct_config.validate()
    pf = preflight_report(archive_dir=archive_dir)
    budget = build_default_budget()

    if pf.verdict != "READY_EXPERIMENTAL":
        # In the real Kaggle notebook, run the M19-C one-cell script here.
        fb = fallback_to_m19c(archive_dir=archive_dir, output_path=output_path, dry_run=True)
        return {
            "mode": "FALLBACK_ONLY",
            "preflight": asdict(pf),
            "fallback": fb,
            "runtime_report": budget.report(),
            "baseline_official_score": PENDING_REAL_ENV,
            "hoct_official_score": PENDING_REAL_ENV,
        }

    # The following block is executed only in the real Kaggle environment.
    # We do not attempt to run it in this sandbox even if a lucky combination
    # of imports succeeded; the preflight verdict is the sole gate.
    return {
        "mode": "EXPERIMENTAL_INTENDED",
        "preflight": asdict(pf),
        "note": ("this scaffold returns without executing detection/HOCT; a real Kaggle "
                 "notebook wraps run() with a phase-instrumented driver that emits the "
                 "runtime report and the shadow comparison"),
        "runtime_report": budget.report(),
        "baseline_official_score": PENDING_REAL_ENV,
        "hoct_official_score": PENDING_REAL_ENV,
    }


__all__ = [
    "DEFAULT_POLICY", "KAGGLE_COMP_DIR", "KAGGLE_REFERENCE_BUNDLE",
    "KAGGLE_WORKING", "KAGGLE_OUTPUT", "REQUIRED_KAGGLE_INPUTS",
    "PreflightReport", "preflight_report",
    "build_default_budget",
    "fallback_to_m19c",
    "run",
]
