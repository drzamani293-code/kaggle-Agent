"""Official-metric vendor/audit CLI scaffold (M39-A).

Goal: pin the competition's official ``tracking_cellmot`` scorer at a specific
commit, vendor every file with SHA-256 recorded, preserve the license, verify
the imports work with Internet disabled, confirm the physical scales (7.0 µm
match distance; z=1.625, y=x=0.40625 voxel scale), and cross-check the
implementation against three synthetic reproduction cases (perfect graph,
one false edge, one false division).

This scaffold does not attempt the audit itself in this environment — the
official source and license file are not present offline here. Instead it
- discovers vendored candidate paths,
- refuses silent fallback to any ``local_metric.py`` file,
- emits ``artifacts/m39_official_metric_audit.json`` populated with
  ``PENDING_REAL_ENV`` in every field that requires the real Kaggle mount
  or a network fetch, and
- returns a non-zero exit code so no downstream code accidentally treats
  the scaffold output as a passing audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import PENDING_REAL_ENV
from .schemas import make_pending_metric_audit, validate_metric_audit


LOCAL_METRIC_PATTERN = re.compile(r"local_metric\.py$")
DEFAULT_VENDOR_ROOTS: tuple[Path, ...] = (
    Path("reference/official_metric"),
    Path("reference/tracking_cellmot"),
    Path("vendor/tracking_cellmot"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_vendored(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        out.append({
            "path": str(p),
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
            "license": p.name.lower() in {"license", "license.txt", "license.md", "copying"},
        })
    return out


def _reject_local_metric_fallback(candidates: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """A vendored copy must NOT contain a ``local_metric.py`` masquerading as
    the official scorer. Return (blocked_ok, offending_paths). ``blocked_ok``
    means we detected no such file OR every such file lives outside the
    vendored root (audit will fail-loud)."""
    offending = [f["path"] for f in candidates if LOCAL_METRIC_PATTERN.search(f["path"])]
    return (not offending), offending


def _internet_disabled() -> bool:
    """Best-effort: return True if we would refuse an internet import path.
    This does not actually attempt DNS; it merely reports whether the
    environment declared itself Internet-off."""
    for flag in ("KAGGLE_INTERNET_DISABLED", "OFFLINE", "NO_NETWORK"):
        if os.environ.get(flag) in {"1", "true", "TRUE"}:
            return True
    # In the Kaggle notebook the default is Internet=Off. The absence of
    # such a flag in the harness sandbox does not prove Internet is on;
    # the audit records PENDING_REAL_ENV for actual verification.
    return False


def audit(comp_dir: Path, vendor_roots: list[Path] | None = None) -> dict[str, Any]:
    d = make_pending_metric_audit()
    roots = vendor_roots or [comp_dir / p for p in DEFAULT_VENDOR_ROOTS]
    vendored: list[dict[str, Any]] = []
    used_root: str | None = None
    for r in roots:
        found = _scan_vendored(r)
        if found:
            vendored = found
            used_root = str(r)
            break

    d["vendored_files"] = vendored
    d["source_repo"] = PENDING_REAL_ENV if not vendored else used_root
    d["license_present"] = any(f["license"] for f in vendored)
    blocked_ok, offending = _reject_local_metric_fallback(vendored)
    d["silent_fallback_to_local_metric_blocked"] = blocked_ok
    if not blocked_ok:
        d["status"] = "REJECTED"
        d["rejection_reason"] = f"local_metric.py detected in vendored roots: {offending}"

    d["internet_disabled_import_ok"] = _internet_disabled()

    # Import verification is intentionally NOT attempted here: this sandbox
    # has no torch and no vendored scorer. If a vendored scorer is later
    # present, the real Kaggle-runner will import it under Internet=Off and
    # populate these fields.
    d["reproduces_edge_jaccard"] = PENDING_REAL_ENV
    d["reproduces_division_jaccard"] = PENDING_REAL_ENV
    d["reproduces_node_penalty"] = PENDING_REAL_ENV
    d["synthetic_case_matches"] = {
        "perfect_graph": PENDING_REAL_ENV,
        "one_false_edge": PENDING_REAL_ENV,
        "one_false_division_missing_daughter": PENDING_REAL_ENV,
    }

    if d.get("status") != "REJECTED":
        d["status"] = PENDING_REAL_ENV
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M39-A official-metric audit scaffold")
    default_comp = Path(__file__).resolve().parents[1]
    parser.add_argument("--comp-dir", default=str(default_comp))
    parser.add_argument(
        "--output",
        default=str(default_comp / "artifacts" / "m39_official_metric_audit.json"),
    )
    ns = parser.parse_args(argv)
    audit_json = audit(Path(ns.comp_dir))
    Path(ns.output).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.output).write_text(json.dumps(audit_json, indent=2, sort_keys=True))
    v = validate_metric_audit(audit_json)
    print(f"WROTE {ns.output}  verdict={v['verdict']}  pending={v['pending_fields']}")
    return 1 if v["verdict"] != "VERIFIED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
