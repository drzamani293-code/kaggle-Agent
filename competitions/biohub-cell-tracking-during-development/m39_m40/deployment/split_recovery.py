"""Exact ``split_0`` recovery cascade — OBSERVED-only.

Mirrors M32.1's approach: recover the training split from an OBSERVED
manifest or checkpoint. Deterministic reconstruction from a seed is
FORBIDDEN because the upstream repo READS ``dataset_splits.json`` (it
never regenerates the split from an observable seed+algorithm — see
``vendor/royerlab_cellmot/PROVENANCE.json``, key ``split_generation``).

Cascade (first match wins):
    A. dataset_splits.json (upstream file) at any known/candidate path
    B. checkpoint metadata via ``torch.load(..., weights_only=True)``,
       looking for keys ``dataset_splits`` / ``splits`` / ``split_0``
    C. explicit split file: ``reference/splits/split_0.json`` or
       ``reference/kaggle_test_splits_50ep.json``
    D. REFUSED  → ``SPLIT_RECONSTRUCTION_FORBIDDEN``

The runner MUST refuse to score anything on the "train" side of split_0
if verdict is anything other than ``OK_OBSERVED`` — no random holdouts,
no seed replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import PENDING_REAL_ENV


# Upstream file that the repo reads (see vendor PROVENANCE); look for it
# by exact filename anywhere under the reference bundle.
UPSTREAM_SPLITS_FILENAME = "dataset_splits.json"

# Additional candidates the operator may ship.
CANDIDATE_EXPLICIT_FILES: tuple[str, ...] = (
    "reference/splits/split_0.json",
    "reference/kaggle_test_splits_50ep.json",
)

# Keys inside a torch checkpoint that we accept as observed split evidence.
CHECKPOINT_SPLIT_KEYS: tuple[str, ...] = (
    "dataset_splits", "splits", "split_0", "split", "train_test_split",
)


@dataclass
class SplitRecoveryResult:
    verdict: str                    # OK_OBSERVED | PENDING_REAL_ENV | REFUSED | ERROR
    split_source: str               # dataset_splits.json | checkpoint | explicit_file | PENDING
    split_source_path: str
    split_source_sha256: str
    train_datasets: list[str] = field(default_factory=list)
    validation_datasets: list[str] = field(default_factory=list)
    overlap: list[str] = field(default_factory=list)
    declared_split_name: str = PENDING_REAL_ENV
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "split_source": self.split_source,
            "split_source_path": self.split_source_path,
            "split_source_sha256": self.split_source_sha256,
            "declared_split_name": self.declared_split_name,
            "train_datasets": list(self.train_datasets),
            "validation_datasets": list(self.validation_datasets),
            "overlap": list(self.overlap),
            "reasons": list(self.reasons),
        }


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_split_json(data: Any) -> tuple[list[str], list[str]]:
    """Accept several shapes emitted by upstream tooling.

    Shapes accepted:
      - {"train": [...], "test": [...]}
      - {"train": [...], "validation": [...]}
      - [{"train": [...], "test": [...]}, ...]  (first element = split_0)
      - {"splits": [...]}                        (first element = split_0)
      - {"split_0": {"train": [...], "test": [...]}}
    """
    if isinstance(data, dict):
        if "split_0" in data and isinstance(data["split_0"], dict):
            return _parse_split_json(data["split_0"])
        if "splits" in data and isinstance(data["splits"], list) and data["splits"]:
            return _parse_split_json(data["splits"][0])
        train = list(data.get("train") or [])
        val = list(data.get("test") or data.get("validation") or data.get("val") or [])
        return train, val
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return _parse_split_json(data[0])
    return [], []


def _scan_for_upstream_splits(root: Path) -> Path | None:
    if not root.exists():
        return None
    hits = sorted(root.rglob(UPSTREAM_SPLITS_FILENAME))
    return hits[0] if hits else None


def _try_checkpoint(ckpt_path: Path) -> tuple[list[str], list[str], str]:
    """Attempt to load a torch checkpoint and pull an observed split.

    Uses ``torch.load(..., weights_only=True)`` when torch is available.
    Returns ``([], [], reason)`` on any failure. NEVER raises.
    """
    try:
        import torch                                                   # type: ignore
    except Exception:                                                    # noqa: BLE001
        return [], [], "torch_not_available"
    try:
        blob = torch.load(str(ckpt_path), weights_only=True,             # type: ignore[arg-type]
                          map_location="cpu")
    except Exception as exc:                                             # noqa: BLE001
        return [], [], f"torch_load_failed: {exc.__class__.__name__}"
    if not isinstance(blob, dict):
        return [], [], "checkpoint_not_dict"
    for k in CHECKPOINT_SPLIT_KEYS:
        if k in blob:
            train, val = _parse_split_json(blob[k])
            if train or val:
                return train, val, f"observed_at_key:{k}"
    return [], [], "no_split_keys_in_checkpoint"


def recover(
    *,
    reference_bundle_root: Path,
    checkpoint_paths: tuple[Path, ...] = (),
) -> SplitRecoveryResult:
    """Run the cascade. ``reference_bundle_root`` is where the upstream
    ``dataset_splits.json`` typically lives inside the Kaggle Dataset.
    ``checkpoint_paths`` is a tuple of ``.pth`` files worth probing."""
    reasons: list[str] = []

    # A. upstream dataset_splits.json
    hit = _scan_for_upstream_splits(reference_bundle_root)
    if hit and hit.is_file():
        try:
            data = json.loads(hit.read_text())
        except json.JSONDecodeError as exc:
            reasons.append(f"upstream_splits_parse_failed:{exc.msg}")
        else:
            train, val = _parse_split_json(data)
            if train or val:
                overlap = sorted(set(train) & set(val))
                return SplitRecoveryResult(
                    verdict="OK_OBSERVED" if not overlap else "REFUSED_LEAKAGE",
                    split_source="dataset_splits.json",
                    split_source_path=str(hit),
                    split_source_sha256=_sha256(hit),
                    train_datasets=train,
                    validation_datasets=val,
                    overlap=overlap,
                    declared_split_name="split_0",
                    reasons=reasons + [f"observed_at:{hit}"],
                )
            reasons.append("upstream_splits_empty")

    # B. checkpoint metadata
    for ckpt in checkpoint_paths:
        if not ckpt.is_file():
            reasons.append(f"checkpoint_missing:{ckpt}")
            continue
        train, val, r = _try_checkpoint(ckpt)
        if train or val:
            overlap = sorted(set(train) & set(val))
            return SplitRecoveryResult(
                verdict="OK_OBSERVED" if not overlap else "REFUSED_LEAKAGE",
                split_source="checkpoint_metadata",
                split_source_path=str(ckpt),
                split_source_sha256=_sha256(ckpt),
                train_datasets=train,
                validation_datasets=val,
                overlap=overlap,
                declared_split_name="split_0",
                reasons=reasons + [r],
            )
        reasons.append(f"checkpoint_no_split:{ckpt}:{r}")

    # C. explicit split files under reference bundle
    for rel in CANDIDATE_EXPLICIT_FILES:
        p = reference_bundle_root / rel
        if p.is_file():
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError as exc:
                reasons.append(f"explicit_split_parse_failed:{p}:{exc.msg}")
                continue
            train, val = _parse_split_json(data)
            if train or val:
                overlap = sorted(set(train) & set(val))
                return SplitRecoveryResult(
                    verdict="OK_OBSERVED" if not overlap else "REFUSED_LEAKAGE",
                    split_source="explicit_split_file",
                    split_source_path=str(p),
                    split_source_sha256=_sha256(p),
                    train_datasets=train,
                    validation_datasets=val,
                    overlap=overlap,
                    declared_split_name="split_0",
                    reasons=reasons + [f"observed_at:{p}"],
                )

    # D. deterministic reconstruction is FORBIDDEN
    return SplitRecoveryResult(
        verdict="REFUSED",
        split_source=PENDING_REAL_ENV,
        split_source_path=PENDING_REAL_ENV,
        split_source_sha256=PENDING_REAL_ENV,
        train_datasets=[],
        validation_datasets=[],
        overlap=[],
        declared_split_name=PENDING_REAL_ENV,
        reasons=reasons + [
            "no observed manifest, no observed checkpoint metadata, "
            "no explicit split file — deterministic reconstruction is forbidden "
            "(upstream repo reads dataset_splits.json; it does not regenerate it)"
        ],
    )


__all__ = [
    "UPSTREAM_SPLITS_FILENAME",
    "CANDIDATE_EXPLICIT_FILES",
    "CHECKPOINT_SPLIT_KEYS",
    "SplitRecoveryResult",
    "recover",
]
