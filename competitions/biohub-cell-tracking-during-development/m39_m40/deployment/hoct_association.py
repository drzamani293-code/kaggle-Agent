"""HOCT association contract (M40-A).

Same shape as ``baseline_association`` but the response object also
records the identical-detection audit. If the HOCT solver mutates the
cached detections in any way, the runner rejects the result and treats
it as a shadow failure (never a submission source).

Real inference is BLOCKED at three points:

1. The M39/M40 preflight — must report ``used_official``
2. This module — refuses to hand back a result if ``detections_reused``
   is False or the pinned config differs from ``HoctAdapterConfig()``
3. The Kaggle runner — refuses to write ``submission.csv`` unless HOCT's
   official score >= baseline's official score on the observed CV
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..hoct_adapter import HoctAdapterConfig, HoctBlockedError


@dataclass
class HoctAssociationRequest:
    dataset: str
    config_id: str
    cached_detections: Any
    dataset_manifest: Mapping[str, Any]
    hoct_config: HoctAdapterConfig = field(default_factory=HoctAdapterConfig)


@dataclass
class HoctAssociationResult:
    dataset: str
    config_id: str
    pred_graph: Any
    n_nodes: int
    n_edges: int
    runtime_seconds: float
    peak_memory_mb: float
    detections_reused: bool
    hoct_config_used: HoctAdapterConfig
    notes: str = ""


HoctSolver = Callable[[HoctAssociationRequest], HoctAssociationResult]


class HoctNotRegisteredError(RuntimeError):
    pass


class HoctIdentityBrokenError(RuntimeError):
    """Raised when HOCT returned a result but ``detections_reused=False``
    (i.e. the baseline vs HOCT comparison would be confounded)."""


_REGISTRY: dict[str, HoctSolver] = {}


def register(name: str, solver: HoctSolver) -> None:
    _REGISTRY[name] = solver


def has(name: str) -> bool:
    return name in _REGISTRY


def get(name: str) -> HoctSolver:
    if name not in _REGISTRY:
        raise HoctNotRegisteredError(
            f"no HOCT solver registered under {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_registered() -> list[str]:
    return sorted(_REGISTRY)


def clear() -> None:
    _REGISTRY.clear()


def run(
    *,
    name: str,
    request: HoctAssociationRequest,
) -> HoctAssociationResult:
    """Look up ``name`` and invoke it. Enforces:

    - the returned ``hoct_config_used`` matches the request's config exactly
      (any deviation -> ``HoctBlockedError``)
    - ``detections_reused`` is True (else ``HoctIdentityBrokenError``)
    """
    # First validate the request's config — max_delta_t=1, n_neighbors<=5, ...
    request.hoct_config.validate()
    solver = get(name)
    result = solver(request)
    if result.hoct_config_used != request.hoct_config:
        raise HoctBlockedError(
            "solver returned a config different from the requested config; "
            f"requested={request.hoct_config!r} used={result.hoct_config_used!r}"
        )
    if not result.detections_reused:
        raise HoctIdentityBrokenError(
            "HOCT solver did NOT reuse the cached baseline detections; "
            "a shadow comparison against baseline is not valid — refusing"
        )
    return result


__all__ = [
    "HoctAssociationRequest", "HoctAssociationResult", "HoctSolver",
    "HoctNotRegisteredError", "HoctIdentityBrokenError",
    "register", "has", "get", "list_registered", "clear", "run",
]
