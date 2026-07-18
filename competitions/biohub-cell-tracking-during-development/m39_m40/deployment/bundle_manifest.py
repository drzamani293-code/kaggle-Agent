"""Bundle-manifest schema and SHA256 verifier.

The offline Kaggle Dataset bundle is a directory tree. Everything the
Kaggle-side runner will ever import, load, or exec must be listed here,
with byte size + SHA256, so the runner can refuse to proceed against a
tampered mount. This module is imported both by the builder (which
computes the manifest) and by the Kaggle-side runner (which verifies it).

Slot categories:

- ``vendor_metric``     : ``vendor/royerlab_cellmot/`` files
- ``m39_m40_package``   : the ``m39_m40/`` scaffolding package
- ``m19c_archive``      : the four M19-C fallback files
- ``hoct_source``       : the vendored official ``hoct/`` package (MIT)
- ``hoct_weights``      : the ``general_v0`` checkpoint (or equivalent)

A slot may be ``STATUS_PRESENT`` or ``STATUS_MISSING``. The runner GATES
on ``STATUS_PRESENT`` for both HOCT slots before allowing the experimental
path; otherwise it emits the M19-C fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .. import PENDING_REAL_ENV


MANIFEST_SCHEMA_VERSION = 1

STATUS_PRESENT = "present"
STATUS_MISSING = "missing"

SLOT_VENDOR_METRIC = "vendor_metric"
SLOT_M39_M40 = "m39_m40_package"
SLOT_M19C = "m19c_archive"
SLOT_HOCT_SOURCE = "hoct_source"
SLOT_HOCT_WEIGHTS = "hoct_weights"

ALL_SLOTS: tuple[str, ...] = (
    SLOT_VENDOR_METRIC,
    SLOT_M39_M40,
    SLOT_M19C,
    SLOT_HOCT_SOURCE,
    SLOT_HOCT_WEIGHTS,
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class FileEntry:
    path: str          # relative to the bundle root
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass
class SlotEntry:
    name: str
    status: str        # STATUS_PRESENT | STATUS_MISSING
    files: list[FileEntry] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "files": [f.to_dict() for f in self.files],
            "notes": self.notes,
        }


@dataclass
class BundleManifest:
    schema_version: int
    bundle_marker: str
    slots: dict[str, SlotEntry]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_marker": self.bundle_marker,
            "provenance": self.provenance,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def _walk_files(root: Path) -> list[Path]:
    """Deterministic sorted walk. Skips __pycache__, .pyc, .DS_Store."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # in-place sort so os.walk yields deterministic order
        dirnames.sort()
        # skip caches
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.endswith(".pyc") or fn == ".DS_Store":
                continue
            out.append(Path(dirpath) / fn)
    return out


def hash_slot(source_root: Path, bundle_root: Path) -> list[FileEntry]:
    """Return FileEntry rows for every file under ``source_root``,
    with paths recorded RELATIVE TO ``bundle_root`` (not to ``source_root``)."""
    if not source_root.exists():
        return []
    entries: list[FileEntry] = []
    for p in _walk_files(source_root):
        rel = p.relative_to(bundle_root)
        entries.append(
            FileEntry(
                path=rel.as_posix(),
                sha256=sha256_file(p),
                bytes=p.stat().st_size,
            )
        )
    return entries


def verify_manifest(bundle_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every entry in ``manifest`` exists under ``bundle_root`` and
    matches SHA256. Missing slots are reported but never mask hash mismatches.

    Returns a report dict with a per-slot verdict and an overall verdict
    ``VERIFIED`` iff every present slot's files match AND both HOCT slots
    are present. Anything else returns ``TAMPERED``, ``INCOMPLETE_HOCT``, or
    ``MISSING_METRIC_OR_ARCHIVE`` — the runner MUST gate on ``VERIFIED``."""
    mismatches: list[dict[str, Any]] = []
    missing_files: list[str] = []
    slot_verdicts: dict[str, str] = {}
    slots = manifest.get("slots", {})

    for slot_name in ALL_SLOTS:
        entry = slots.get(slot_name)
        if entry is None:
            slot_verdicts[slot_name] = "slot_absent_from_manifest"
            continue
        if entry.get("status") == STATUS_MISSING:
            slot_verdicts[slot_name] = STATUS_MISSING
            continue

        slot_ok = True
        for f in entry.get("files", []):
            fp = bundle_root / f["path"]
            if not fp.is_file():
                missing_files.append(f["path"])
                slot_ok = False
                continue
            actual = sha256_file(fp)
            if actual != f["sha256"]:
                mismatches.append(
                    {"path": f["path"], "expected": f["sha256"], "actual": actual}
                )
                slot_ok = False
        slot_verdicts[slot_name] = STATUS_PRESENT if slot_ok else "tampered"

    def _present(slot_name: str) -> bool:
        return slot_verdicts.get(slot_name) == STATUS_PRESENT

    if mismatches or any(v == "tampered" for v in slot_verdicts.values()):
        overall = "TAMPERED"
    elif not _present(SLOT_VENDOR_METRIC) or not _present(SLOT_M19C) or not _present(SLOT_M39_M40):
        overall = "MISSING_METRIC_OR_ARCHIVE"
    elif not _present(SLOT_HOCT_SOURCE) or not _present(SLOT_HOCT_WEIGHTS):
        overall = "INCOMPLETE_HOCT"
    else:
        overall = "VERIFIED"

    return {
        "schema_version": manifest.get("schema_version"),
        "bundle_root": str(bundle_root),
        "slot_verdicts": slot_verdicts,
        "mismatches": mismatches,
        "missing_files": missing_files,
        "verdict": overall,
    }


def load_manifest(bundle_root: Path, filename: str) -> dict[str, Any]:
    path = bundle_root / filename
    if not path.is_file():
        return {"schema_version": None, "bundle_marker": PENDING_REAL_ENV,
                "slots": {}, "error": "manifest_file_missing"}
    return json.loads(path.read_text())


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "STATUS_PRESENT", "STATUS_MISSING",
    "SLOT_VENDOR_METRIC", "SLOT_M39_M40", "SLOT_M19C",
    "SLOT_HOCT_SOURCE", "SLOT_HOCT_WEIGHTS", "ALL_SLOTS",
    "FileEntry", "SlotEntry", "BundleManifest",
    "sha256_file", "hash_slot", "verify_manifest", "load_manifest",
]
