"""Split-audit CLI scaffold (M39-B).

Recovers the split that the controlled ``split_0`` model was trained on, from
manifest files or checkpoint metadata. Refuses to reconstruct a split from an
unobserved random seed. If the exact ``split_0`` cannot be proven, this
scaffold instead labels the produced split ``diagnostic_leave_one_out_*`` and
records that fact.

Writes:
- ``artifacts/m39_split_audit.json``
- ``artifacts/m39_split_membership.csv``

In the current sandbox there is no reference bundle or checkpoint metadata
mounted; the produced JSON is populated with ``PENDING_REAL_ENV`` for every
field that requires the real Kaggle mount. Never invent membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from . import PENDING_REAL_ENV
from .schemas import SPLIT_AUDIT_KEYS, validate_split_audit


CANDIDATE_MANIFEST_PATHS: tuple[str, ...] = (
    "reference/kaggle_test_splits_50ep.json",
    "reference/splits/split_0.json",
    "reference/manifests/split_membership.csv",
    "artifacts/m39_split_membership.csv",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _empty_audit(reason: str) -> dict[str, Any]:
    d: dict[str, Any] = {k: PENDING_REAL_ENV for k in SPLIT_AUDIT_KEYS}
    d["reconstruction_from_seed"] = False
    d["train_datasets"] = []
    d["validation_datasets"] = []
    d["train_validation_overlap"] = []
    d["notes"] = reason
    return d


def _load_membership_csv(path: Path) -> tuple[list[str], list[str]]:
    train: list[str] = []
    val: list[str] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "dataset" not in reader.fieldnames \
                or "split_role" not in reader.fieldnames:
            return train, val
        for r in reader:
            role = (r.get("split_role") or "").strip().lower()
            ds = (r.get("dataset") or "").strip()
            if not ds:
                continue
            if role == "train":
                train.append(ds)
            elif role in {"val", "validation"}:
                val.append(ds)
    return train, val


def audit(comp_dir: Path, membership_out: Path) -> dict[str, Any]:
    candidates = [comp_dir / p for p in CANDIDATE_MANIFEST_PATHS]
    src = _first_existing(candidates)
    if src is None:
        d = _empty_audit(
            "no manifest, checkpoint metadata, or explicit split file found "
            "under reference/ or artifacts/; cannot prove split_0 offline"
        )
        d["split_source"] = PENDING_REAL_ENV
        d["split_source_path"] = PENDING_REAL_ENV
        d["split_source_sha256"] = PENDING_REAL_ENV
        d["declared_split_name"] = PENDING_REAL_ENV
        d["membership_csv_path"] = str(membership_out.relative_to(comp_dir)) \
            if membership_out.is_relative_to(comp_dir) else str(membership_out)
        d["membership_csv_sha256"] = PENDING_REAL_ENV
        d["status"] = PENDING_REAL_ENV
        return d

    train: list[str] = []
    val: list[str] = []
    if src.suffix.lower() == ".csv":
        train, val = _load_membership_csv(src)
    elif src.suffix.lower() == ".json":
        try:
            data = json.loads(src.read_text())
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            train = list(data[0].get("train") or [])
            val = list(data[0].get("test") or [])   # audited splits use "test"
        elif isinstance(data, dict):
            train = list(data.get("train") or [])
            val = list(data.get("test") or data.get("validation") or [])

    overlap = sorted(set(train) & set(val))
    membership_out.parent.mkdir(parents=True, exist_ok=True)
    with open(membership_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split_role"])
        for ds in train:
            w.writerow([ds, "train"])
        for ds in val:
            w.writerow([ds, "validation"])

    proven = src.name in {"kaggle_test_splits_50ep.json", "split_0.json"}
    d: dict[str, Any] = _empty_audit(
        "membership derived from an explicit manifest file"
    )
    d["split_source"] = "manifest" if proven else "diagnostic_LODO"
    d["split_source_path"] = str(src)
    d["split_source_sha256"] = _sha256(src)
    d["declared_split_name"] = "split_0" if proven else "diagnostic_LODO_from_" + src.name
    d["reconstruction_from_seed"] = False
    d["train_datasets"] = train
    d["validation_datasets"] = val
    d["train_validation_overlap"] = overlap
    d["membership_csv_path"] = str(membership_out)
    d["membership_csv_sha256"] = _sha256(membership_out)
    d["status"] = "verified" if proven and not overlap else "diagnostic"
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M39-B split audit scaffold")
    default_comp = Path(__file__).resolve().parents[1]
    parser.add_argument("--comp-dir", default=str(default_comp))
    parser.add_argument(
        "--audit-out",
        default=str(default_comp / "artifacts" / "m39_split_audit.json"),
    )
    parser.add_argument(
        "--membership-out",
        default=str(default_comp / "artifacts" / "m39_split_membership.csv"),
    )
    ns = parser.parse_args(argv)
    audit_json = audit(Path(ns.comp_dir), Path(ns.membership_out))
    Path(ns.audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.audit_out).write_text(json.dumps(audit_json, indent=2, sort_keys=True))
    v = validate_split_audit(audit_json)
    print(f"WROTE {ns.audit_out}  verdict={v['verdict']}  "
          f"leakage_detected={v['leakage_detected']}  "
          f"reconstructed={v['reconstructed_from_seed']}")
    return 0 if v["verdict"] in {"OK_PROVEN", "OK_DIAGNOSTIC"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
