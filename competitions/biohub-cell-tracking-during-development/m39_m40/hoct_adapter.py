"""M40-A HOCT shadow adapter (schema-only, real inference BLOCKED).

Design constraints (from the M39/M40 master prompt):

1. Keep detections IDENTICAL to the current baseline; only replace the
   association engine.
2. Never blindly call HOCT's ``create_graph_from_points``: in the inspected
   public release it may be a stub. This adapter must inspect the installed
   source and refuse to call a stub.
3. Runtime-first knobs are pinned: ``max_delta_t=1``, ``n_neighbors=3``
   (at most 5), no HOCT TTA, no tiling unless memory requires it, no long-gap
   pass, strict ILP time limit.
4. Adapter kinds allowed:
   (a) preferred: label masks / local pseudo-masks + HOCT standard graph
       feature extraction,
   (b) alternative: point-to-graph adapter that produces the exact node/edge
       schema and required standardized features (extra schema tests).

In this sandbox no HOCT source is vendored, no model weights are mounted, no
torch/GPU is available, and no image data is present. Every entry point in
this module that would perform real inference RAISES ``HoctBlockedError``
carrying the precise reason. Only the schema helpers are executable.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import PENDING_REAL_ENV


# --------------------------------------------------------------------------- #
# Adapter configuration & required schema
# --------------------------------------------------------------------------- #

MAX_DELTA_T = 1
MAX_N_NEIGHBORS = 5
DEFAULT_N_NEIGHBORS = 3

HOCT_NODE_FIELDS: tuple[str, ...] = (
    "node_id",
    "dataset",
    "t",
    "z_um", "y_um", "x_um",       # physical micrometres
    "z_vox", "y_vox", "x_vox",     # voxel coords
    "detection_score",
    "mask_label",                  # -1 if using point-only alternative
)
"""Node schema every HOCT-adapter row must carry."""

HOCT_EDGE_FIELDS: tuple[str, ...] = (
    "edge_id",
    "dataset",
    "source_id",
    "target_id",
    "source_t",
    "target_t",
    "delta_t",                     # target_t - source_t, must be == 1
    "displacement_um",
)
"""Edge schema every HOCT-adapter candidate row must carry."""

HOCT_REQUIRED_NODE_FEATURES: tuple[str, ...] = (
    "detection_score",
    "z_um", "y_um", "x_um",
    "mask_area_vox",
    "mask_intensity_mean",
    "mask_intensity_std",
)
"""Standardized features HOCT expects on every node."""

HOCT_REQUIRED_EDGE_FEATURES: tuple[str, ...] = (
    "displacement_um",
    "delta_t",
    "source_detection_score",
    "target_detection_score",
)
"""Standardized features HOCT expects on every candidate edge."""


@dataclass(frozen=True)
class HoctAdapterConfig:
    """Runtime-first configuration; runtime-safety fields are pinned."""
    kind: str = "label_masks"                  # or "point_to_graph"
    max_delta_t: int = MAX_DELTA_T
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    use_tta: bool = False
    use_tiling: bool = False
    use_long_gap_pass: bool = False
    ilp_time_limit_s: float = 60.0
    inference_mode: bool = True
    dtype: str = "float32"

    def validate(self) -> None:
        if self.kind not in {"label_masks", "point_to_graph"}:
            raise ValueError(f"unknown adapter kind: {self.kind!r}")
        if self.max_delta_t != MAX_DELTA_T:
            raise ValueError(
                f"M40-A requires max_delta_t={MAX_DELTA_T}; got {self.max_delta_t}"
            )
        if not (1 <= self.n_neighbors <= MAX_N_NEIGHBORS):
            raise ValueError(
                f"n_neighbors must be in [1, {MAX_N_NEIGHBORS}]; got {self.n_neighbors}"
            )
        if self.use_tta or self.use_long_gap_pass:
            raise ValueError("M40-A forbids HOCT TTA and long-gap passes")
        if self.ilp_time_limit_s <= 0:
            raise ValueError("ILP time limit must be strictly positive")


class HoctBlockedError(RuntimeError):
    """Raised whenever an adapter entry point would perform real inference
    without the prerequisites (torch, GPU, HOCT source, weights, images,
    features) actually present."""


# --------------------------------------------------------------------------- #
# Schema helpers (safe, no runtime deps)
# --------------------------------------------------------------------------- #

def validate_node_schema(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every row must contain every HOCT_NODE_FIELDS key; every
    HOCT_REQUIRED_NODE_FEATURES key must be present too."""
    missing_fields: set[str] = set()
    missing_features: set[str] = set()
    n = 0
    for r in rows:
        n += 1
        missing_fields.update(k for k in HOCT_NODE_FIELDS if k not in r)
        missing_features.update(k for k in HOCT_REQUIRED_NODE_FEATURES if k not in r)
    return {
        "n_rows": n,
        "missing_fields": sorted(missing_fields),
        "missing_features": sorted(missing_features),
        "ok": n > 0 and not missing_fields and not missing_features,
    }


def validate_edge_schema(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    missing_fields: set[str] = set()
    missing_features: set[str] = set()
    delta_t_ok = True
    time_order_ok = True
    n = 0
    for r in rows:
        n += 1
        missing_fields.update(k for k in HOCT_EDGE_FIELDS if k not in r)
        missing_features.update(k for k in HOCT_REQUIRED_EDGE_FEATURES if k not in r)
        try:
            dt = int(r["delta_t"])
            st, tt = int(r["source_t"]), int(r["target_t"])
            if dt != MAX_DELTA_T:
                delta_t_ok = False
            if not (st < tt):
                time_order_ok = False
        except (KeyError, TypeError, ValueError):
            delta_t_ok = False
            time_order_ok = False
    return {
        "n_rows": n,
        "missing_fields": sorted(missing_fields),
        "missing_features": sorted(missing_features),
        "delta_t_ok": delta_t_ok,
        "time_order_ok": time_order_ok,
        "ok": (n > 0 and not missing_fields and not missing_features
               and delta_t_ok and time_order_ok),
    }


# --------------------------------------------------------------------------- #
# HOCT source / stub detection
# --------------------------------------------------------------------------- #

def probe_hoct_source() -> dict[str, Any]:
    """Detect whether the installed/vendored ``hoct`` package exposes a
    non-stub ``create_graph_from_points`` and record its source location.
    Never imports anything else from HOCT."""
    spec = importlib.util.find_spec("hoct")
    if spec is None:
        return {
            "hoct_installed": False,
            "create_graph_from_points_status": PENDING_REAL_ENV,
            "notes": "hoct package not importable in this environment",
        }
    try:
        import hoct  # type: ignore  # noqa: F401
        graph_mod = importlib.import_module("hoct.graph") \
            if importlib.util.find_spec("hoct.graph") else hoct
    except Exception as exc:                                # noqa: BLE001
        return {
            "hoct_installed": True,
            "create_graph_from_points_status": "import_failed",
            "notes": f"{type(exc).__name__}: {exc}",
        }
    fn = getattr(graph_mod, "create_graph_from_points", None)
    if fn is None:
        return {
            "hoct_installed": True,
            "create_graph_from_points_status": "missing",
            "notes": "hoct is installed but exposes no create_graph_from_points",
        }
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = ""
    is_stub = _looks_like_stub(src)
    return {
        "hoct_installed": True,
        "create_graph_from_points_status": "stub_detected_blocked" if is_stub else "used_official",
        "source_file": inspect.getfile(fn) if callable(fn) else None,
        "source_length": len(src),
        "notes": "stub heuristic: body contains only 'pass' / 'raise NotImplementedError'"
                 " / 'return None'" if is_stub else "",
    }


_STUB_MARKERS = (
    "raise NotImplementedError",
    "return None",
)


def _looks_like_stub(src: str) -> bool:
    stripped = "\n".join(
        ln for ln in src.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    if not stripped:
        return True
    body_lines = [ln for ln in stripped.splitlines() if not ln.lstrip().startswith("def ")]
    non_docstring = "\n".join(
        ln for ln in body_lines
        if not (ln.strip().startswith(("'", '"')))
    ).strip()
    if not non_docstring:
        return True
    if non_docstring in {"pass"}:
        return True
    if any(m in non_docstring for m in _STUB_MARKERS) and len(non_docstring.splitlines()) <= 3:
        return True
    return False


# --------------------------------------------------------------------------- #
# Prerequisite checks (torch/GPU/weights/images/features)
# --------------------------------------------------------------------------- #

@dataclass
class HoctPrereqs:
    torch_importable: bool
    cuda_available: bool
    hoct_probe: dict[str, Any]
    weight_path: Path | None
    weight_present: bool
    images_dir: Path | None
    images_present: bool
    baseline_detections_path: Path | None
    baseline_detections_present: bool
    missing: list[str] = field(default_factory=list)


def check_prereqs(
    *,
    weight_path: Path | None = None,
    images_dir: Path | None = None,
    baseline_detections_path: Path | None = None,
) -> HoctPrereqs:
    torch_ok = importlib.util.find_spec("torch") is not None
    cuda_ok = False
    if torch_ok:
        try:
            import torch  # type: ignore
            cuda_ok = bool(torch.cuda.is_available())      # type: ignore[attr-defined]
        except Exception:                                    # noqa: BLE001
            cuda_ok = False
    probe = probe_hoct_source()
    weight_present = bool(weight_path and weight_path.exists())
    images_present = bool(images_dir and images_dir.is_dir())
    detections_present = bool(baseline_detections_path and baseline_detections_path.exists())
    missing: list[str] = []
    if not torch_ok:
        missing.append("torch")
    if not cuda_ok:
        missing.append("cuda_gpu")
    if probe.get("create_graph_from_points_status") not in {"used_official"}:
        missing.append("hoct_create_graph_from_points_or_valid_adapter")
    if not weight_present:
        missing.append("hoct_weight_file")
    if not images_present:
        missing.append("images_dir")
    if not detections_present:
        missing.append("baseline_detections")
    return HoctPrereqs(
        torch_importable=torch_ok,
        cuda_available=cuda_ok,
        hoct_probe=probe,
        weight_path=weight_path,
        weight_present=weight_present,
        images_dir=images_dir,
        images_present=images_present,
        baseline_detections_path=baseline_detections_path,
        baseline_detections_present=detections_present,
        missing=missing,
    )


# --------------------------------------------------------------------------- #
# Blocked entry points (real inference must go via the Kaggle runner)
# --------------------------------------------------------------------------- #

def build_hoct_graph(
    *,
    detections: Any,
    config: HoctAdapterConfig,
    prereqs: HoctPrereqs,
) -> Any:
    """Build the HOCT candidate graph from the baseline detections. RAISES
    ``HoctBlockedError`` if any prerequisite is missing OR if the installed
    HOCT ``create_graph_from_points`` is a stub."""
    config.validate()
    if prereqs.missing:
        raise HoctBlockedError(
            f"HOCT graph construction blocked; missing prerequisites: {prereqs.missing}"
        )
    if prereqs.hoct_probe.get("create_graph_from_points_status") == "stub_detected_blocked":
        raise HoctBlockedError(
            "HOCT create_graph_from_points is a stub in the installed source; refuse to call. "
            "Use the point_to_graph vendored adapter instead."
        )
    raise HoctBlockedError(
        "build_hoct_graph is a scaffold. Real graph construction is only permitted from the "
        "Kaggle runner after check_prereqs() reports no missing items and probe_hoct_source() "
        "reports 'used_official'."
    )


def run_shadow_inference(*args: Any, **kwargs: Any) -> None:
    raise HoctBlockedError(
        "run_shadow_inference is a scaffold and must never run in this sandbox: no torch, "
        "no GPU, no vendored HOCT, no weights, no image data. Real inference is driven by "
        "m39_m40.kaggle_runner in the Kaggle GPU notebook."
    )


__all__ = [
    "MAX_DELTA_T", "MAX_N_NEIGHBORS", "DEFAULT_N_NEIGHBORS",
    "HOCT_NODE_FIELDS", "HOCT_EDGE_FIELDS",
    "HOCT_REQUIRED_NODE_FEATURES", "HOCT_REQUIRED_EDGE_FEATURES",
    "HoctAdapterConfig", "HoctBlockedError", "HoctPrereqs",
    "validate_node_schema", "validate_edge_schema",
    "probe_hoct_source", "check_prereqs",
    "build_hoct_graph", "run_shadow_inference",
]
