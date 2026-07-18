"""M39/M40 scaffolding.

This package is deliberately GPU-free and Kaggle-mount-free. It contains:

- `runtime_guard` : wall-clock runtime governor (vendored from
  `biohub_runtime_guard.py` in the M39/M40 start pack).
- `schemas`      : dataclass/JSON schemas for the CV result rows, CV summary,
  runtime report, official-metric audit, split audit, and HOCT shadow report.
- `inventory`    : Step-0 repository inventory generator (M39).
- `metric_vendor`: official-metric vendor/audit CLI scaffold (M39-A).
- `split_audit`  : split-audit CLI scaffold (M39-B).
- `offline_check`: offline dependency import check.
- `hoct_adapter` : HOCT schema-only adapter (M40-A). Blocks real inference
  when torch, GPU, weights, image data, or required features are absent.
- `kaggle_runner`: Kaggle real-environment runner/checklist scaffold that
  will later enforce the 105/110-minute budgets, preserve the 10-minute
  finalization reserve, and emit an M19-C-equivalent fallback if the
  experimental path cannot finish safely.

Every CV score, GPU runtime, model hash, and successful-import claim that
requires the real Kaggle environment is emitted with the sentinel value
``PENDING_REAL_ENV`` in this package. Never fabricate.
"""

PENDING_REAL_ENV = "PENDING_REAL_ENV"
"""Marker for values that MUST be produced by the real Kaggle environment.

Any consumer that receives this value must refuse to proceed with a
submission decision. Do not overwrite this string with a fabricated number.
"""
