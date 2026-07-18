"""Baseline association contract (M39-C).

The Kaggle runner needs a callable that, given cached detections + a
config id, returns a predicted graph in the format the official
``tracking_cellmot`` metric consumes. The concrete implementation lives
inside the reference-0902 notebook that runs inside the Kaggle
environment; this module defines the CONTRACT and a REGISTRY that the
runner uses to look up a registered implementation.

If nothing is registered by the time the runner asks for the baseline,
``BaselineNotRegisteredError`` is raised and the runner falls back to
M19-C. No fake baseline is ever synthesised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass
class BaselineAssociationRequest:
    dataset: str
    config_id: str
    cached_detections: Any
    dataset_manifest: Mapping[str, Any]


@dataclass
class BaselineAssociationResult:
    dataset: str
    config_id: str
    pred_graph: Any                        # tracksdata graph in metric-native format
    n_nodes: int
    n_edges: int
    runtime_seconds: float
    peak_memory_mb: float
    detections_reused: bool                # True iff cached_detections were consumed unchanged
    notes: str = ""


BaselineSolver = Callable[[BaselineAssociationRequest], BaselineAssociationResult]


class BaselineNotRegisteredError(RuntimeError):
    pass


_REGISTRY: dict[str, BaselineSolver] = {}


def register(name: str, solver: BaselineSolver) -> None:
    """Register a baseline solver under a unique name (e.g. ``m19c``,
    ``reference_0902_current_kernel``). Duplicate names OVERWRITE."""
    _REGISTRY[name] = solver


def has(name: str) -> bool:
    return name in _REGISTRY


def get(name: str) -> BaselineSolver:
    if name not in _REGISTRY:
        raise BaselineNotRegisteredError(
            f"no baseline solver registered under {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_registered() -> list[str]:
    return sorted(_REGISTRY)


def clear() -> None:
    _REGISTRY.clear()


__all__ = [
    "BaselineAssociationRequest",
    "BaselineAssociationResult",
    "BaselineSolver",
    "BaselineNotRegisteredError",
    "register", "has", "get", "list_registered", "clear",
]
