"""Offline Kaggle Dataset bundle builder.

Assembles the M39/M40 bundle FROM the current repository INTO a chosen
output directory. The output directory is meant to be uploaded verbatim
as a Kaggle Dataset. It contains:

    <bundle_root>/
      M39_M40_BUNDLE.marker              # discovery beacon
      BUNDLE_MANIFEST.json               # per-file SHA256, per-slot status
      vendor/royerlab_cellmot/           # official metric (byte-identical)
      m39_m40/                           # M39/M40 scaffolding
      archive/m19c/                      # immutable fallback
      hoct/                              # OPTIONAL — official MIT source
      weights/hoct/general_v0/           # OPTIONAL — general_v0 checkpoint

By default the builder REFUSES to complete without the HOCT source and
weights and reports ``INCOMPLETE_HOCT``. Pass ``--allow-hoct-missing`` to
build a partial bundle for the operator to top up before upload.

Provenance recorded per slot:
    - upstream URL / pinned commit (from vendored PROVENANCE if available)
    - byte count and SHA256 for every file
    - git HEAD of THIS repository at build time
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import PENDING_REAL_ENV
from . import BUNDLE_MANIFEST_FILENAME, BUNDLE_MARKER_FILENAME
from .bundle_manifest import (
    BundleManifest,
    MANIFEST_SCHEMA_VERSION,
    SLOT_HOCT_SOURCE,
    SLOT_HOCT_WEIGHTS,
    SLOT_M19C,
    SLOT_M39_M40,
    SLOT_VENDOR_METRIC,
    STATUS_MISSING,
    STATUS_PRESENT,
    SlotEntry,
    hash_slot,
)


DEFAULT_COMP_DIR = Path(__file__).resolve().parents[2]


def _git_head(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:                                                # noqa: BLE001
        return PENDING_REAL_ENV


def _copy_tree(src: Path, dst: Path, *, skip_pycache: bool = True) -> int:
    """Copy ``src`` -> ``dst`` recursively. Returns file count."""
    if not src.exists():
        return 0
    count = 0
    for p in sorted(src.rglob("*")):
        if p.is_dir():
            continue
        if skip_pycache and (".pyc" in p.name or "__pycache__" in p.parts):
            continue
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        count += 1
    return count


def _vendor_metric_provenance(vendor_root: Path) -> dict[str, Any]:
    prov = vendor_root / "PROVENANCE.json"
    if not prov.is_file():
        return {"error": "PROVENANCE.json missing"}
    try:
        return json.loads(prov.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"PROVENANCE.json parse failed: {exc}"}


def _hoct_provenance(hoct_root: Path) -> dict[str, Any]:
    """Read a HOCT provenance file if present. The operator must ship a
    ``PROVENANCE.json`` next to the vendored ``hoct/`` package containing
    ``upstream_repository``, ``upstream_commit``, ``license``,
    ``vendored_at``. Missing = ``PENDING_REAL_ENV``."""
    prov = hoct_root / "PROVENANCE.json"
    if not prov.is_file():
        return {
            "upstream_repository": PENDING_REAL_ENV,
            "upstream_commit": PENDING_REAL_ENV,
            "license": PENDING_REAL_ENV,
            "note": "HOCT PROVENANCE.json missing; operator must vendor it "
                    "with upstream_repository/commit/license fields before upload",
        }
    try:
        return json.loads(prov.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"HOCT PROVENANCE.json parse failed: {exc}"}


def build(
    *,
    comp_dir: Path,
    bundle_root: Path,
    hoct_source: Path | None,
    hoct_weights: Path | None,
    allow_hoct_missing: bool = False,
    marker_note: str = "",
) -> dict[str, Any]:
    """Materialise the bundle. Returns a summary dict.

    Raises FileNotFoundError if a required non-HOCT slot is missing (the
    tracking_cellmot vendor, the m39_m40 package, or the M19-C archive)."""
    bundle_root.mkdir(parents=True, exist_ok=True)

    # ---- required source slots -------------------------------------------- #
    vendor_src = comp_dir / "vendor" / "royerlab_cellmot"
    if not vendor_src.is_dir():
        raise FileNotFoundError(
            f"required slot missing: vendor/royerlab_cellmot under {comp_dir}"
        )
    m39_src = comp_dir / "m39_m40"
    if not m39_src.is_dir():
        raise FileNotFoundError(f"required slot missing: m39_m40 under {comp_dir}")
    m19c_src = comp_dir / "archive" / "m19c"
    if not m19c_src.is_dir():
        raise FileNotFoundError(f"required slot missing: archive/m19c under {comp_dir}")

    # ---- copy required slots --------------------------------------------- #
    vendor_dst = bundle_root / "vendor" / "royerlab_cellmot"
    m39_dst = bundle_root / "m39_m40"
    m19c_dst = bundle_root / "archive" / "m19c"

    n_vendor = _copy_tree(vendor_src, vendor_dst)
    n_m39 = _copy_tree(m39_src, m39_dst)
    n_m19c = _copy_tree(m19c_src, m19c_dst)

    # ---- optional HOCT slots --------------------------------------------- #
    hoct_dst = bundle_root / "hoct"
    weights_dst = bundle_root / "weights" / "hoct" / "general_v0"
    n_hoct_src = _copy_tree(hoct_source, hoct_dst) if hoct_source else 0
    n_hoct_wts = _copy_tree(hoct_weights, weights_dst) if hoct_weights else 0

    # ---- build manifest --------------------------------------------------- #
    slots: dict[str, SlotEntry] = {}
    slots[SLOT_VENDOR_METRIC] = SlotEntry(
        name=SLOT_VENDOR_METRIC,
        status=STATUS_PRESENT if n_vendor else STATUS_MISSING,
        files=hash_slot(vendor_dst, bundle_root),
        notes=f"{n_vendor} files copied from {vendor_src}",
    )
    slots[SLOT_M39_M40] = SlotEntry(
        name=SLOT_M39_M40,
        status=STATUS_PRESENT if n_m39 else STATUS_MISSING,
        files=hash_slot(m39_dst, bundle_root),
        notes=f"{n_m39} files copied from {m39_src}",
    )
    slots[SLOT_M19C] = SlotEntry(
        name=SLOT_M19C,
        status=STATUS_PRESENT if n_m19c else STATUS_MISSING,
        files=hash_slot(m19c_dst, bundle_root),
        notes=f"{n_m19c} files copied from {m19c_src}",
    )
    slots[SLOT_HOCT_SOURCE] = SlotEntry(
        name=SLOT_HOCT_SOURCE,
        status=STATUS_PRESENT if n_hoct_src else STATUS_MISSING,
        files=hash_slot(hoct_dst, bundle_root) if n_hoct_src else [],
        notes=(f"{n_hoct_src} files copied from {hoct_source}"
               if hoct_source else "no --hoct-source path supplied"),
    )
    slots[SLOT_HOCT_WEIGHTS] = SlotEntry(
        name=SLOT_HOCT_WEIGHTS,
        status=STATUS_PRESENT if n_hoct_wts else STATUS_MISSING,
        files=hash_slot(weights_dst, bundle_root) if n_hoct_wts else [],
        notes=(f"{n_hoct_wts} weight files copied from {hoct_weights}"
               if hoct_weights else "no --hoct-weights path supplied"),
    )

    manifest = BundleManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        bundle_marker=BUNDLE_MARKER_FILENAME,
        provenance={
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "kaggle_agent_repo_head": _git_head(comp_dir.parents[1] if len(comp_dir.parents) > 1 else comp_dir),
            "vendor_metric_provenance": _vendor_metric_provenance(vendor_src),
            "hoct_provenance": _hoct_provenance(hoct_source) if hoct_source else {
                "upstream_repository": PENDING_REAL_ENV,
                "upstream_commit": PENDING_REAL_ENV,
                "license": PENDING_REAL_ENV,
                "note": "no --hoct-source path supplied at build time",
            },
            "hoct_weights_source": str(hoct_weights) if hoct_weights else PENDING_REAL_ENV,
            "marker_note": marker_note,
        },
        slots=slots,
    )
    manifest.write(bundle_root / BUNDLE_MANIFEST_FILENAME)

    # Marker file — non-empty so an accidental empty file won't count.
    marker = bundle_root / BUNDLE_MARKER_FILENAME
    marker.write_text(
        "M39_M40 offline bundle\n"
        f"schema_version={MANIFEST_SCHEMA_VERSION}\n"
        f"note={marker_note}\n"
    )

    # ---- verdict --------------------------------------------------------- #
    hoct_ok = slots[SLOT_HOCT_SOURCE].status == STATUS_PRESENT \
        and slots[SLOT_HOCT_WEIGHTS].status == STATUS_PRESENT
    if hoct_ok:
        verdict = "COMPLETE"
    elif allow_hoct_missing:
        verdict = "PARTIAL_HOCT_MISSING"
    else:
        # Refuse to declare the bundle ready. The bundle files are still on
        # disk (useful for the operator to inspect) but the summary flags it.
        verdict = "REFUSED_HOCT_MISSING"

    return {
        "verdict": verdict,
        "bundle_root": str(bundle_root),
        "file_counts": {
            SLOT_VENDOR_METRIC: n_vendor,
            SLOT_M39_M40: n_m39,
            SLOT_M19C: n_m19c,
            SLOT_HOCT_SOURCE: n_hoct_src,
            SLOT_HOCT_WEIGHTS: n_hoct_wts,
        },
        "manifest_path": str(bundle_root / BUNDLE_MANIFEST_FILENAME),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the offline M39/M40 Kaggle Dataset bundle.")
    parser.add_argument("--comp-dir", default=str(DEFAULT_COMP_DIR),
                        help="path to the competitions/biohub-cell-tracking-during-development directory")
    parser.add_argument("--bundle-root", required=True,
                        help="output directory (uploaded as a Kaggle Dataset)")
    parser.add_argument("--hoct-source", default=None,
                        help="path to the vendored official MIT-licensed hoct/ package")
    parser.add_argument("--hoct-weights", default=None,
                        help="path to the general_v0 checkpoint directory")
    parser.add_argument("--allow-hoct-missing", action="store_true",
                        help="build a partial bundle even when HOCT is absent")
    parser.add_argument("--marker-note", default="",
                        help="short human note recorded in the marker file and manifest")
    ns = parser.parse_args(argv)

    try:
        result = build(
            comp_dir=Path(ns.comp_dir).resolve(),
            bundle_root=Path(ns.bundle_root).resolve(),
            hoct_source=Path(ns.hoct_source).resolve() if ns.hoct_source else None,
            hoct_weights=Path(ns.hoct_weights).resolve() if ns.hoct_weights else None,
            allow_hoct_missing=ns.allow_hoct_missing,
            marker_note=ns.marker_note,
        )
    except FileNotFoundError as exc:
        print(f"BUILD_FAILED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] == "COMPLETE":
        return 0
    if result["verdict"] == "PARTIAL_HOCT_MISSING":
        return 0            # explicitly opted-in
    return 3                # REFUSED_HOCT_MISSING


if __name__ == "__main__":
    raise SystemExit(main())
