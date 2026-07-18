"""M39/M40 Kaggle deployment package.

The ``deployment`` subpackage assembles a byte-verified offline Kaggle
Dataset bundle from the vendored official metric, the M39/M40 scaffolding,
the M19-C immutable archive, and (when supplied by the operator) the
official HOCT source + ``general_v0`` checkpoint. It also carries the
Kaggle-side runner that verifies the mounted bundle and drives the full
M39 official CV → M40-A HOCT shadow → validate → M19-C fallback pipeline
under the runtime governor.

Every value that requires the real Kaggle GPU environment stays
``PENDING_REAL_ENV`` until the runner writes it from a real Kaggle cell.
"""

from __future__ import annotations

from .. import PENDING_REAL_ENV  # re-export for convenience

BUNDLE_MARKER_FILENAME = "M39_M40_BUNDLE.marker"
BUNDLE_MANIFEST_FILENAME = "BUNDLE_MANIFEST.json"

__all__ = ["PENDING_REAL_ENV", "BUNDLE_MARKER_FILENAME", "BUNDLE_MANIFEST_FILENAME"]
