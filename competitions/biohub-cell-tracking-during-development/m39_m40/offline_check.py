"""Offline dependency import check.

Emits a JSON report of which project-critical modules are importable in the
current environment, WITHOUT attempting any network activity. Every module
that fails to import is reported honestly — never fabricated.

Used by the M39 official-CV harness and the M40-A HOCT adapter to refuse
to proceed when a required dependency is missing.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any


PROJECT_MODULES: tuple[str, ...] = (
    # numerical / graph
    "numpy",
    "pandas",
    "scipy",
    "scipy.optimize",
    "networkx",
    # image / zarr
    "zarr",
    "tifffile",
    "skimage",
    # ML
    "torch",
    "torchvision",
    # tracking / metric candidates
    "tracksdata",
    "geff",
    "tracking_cellmot",
    "tracking_cellmot.metrics",
    # official metric alt paths
    "royerlab_tracking",
    "hoct",
    # ILP
    "gurobipy",
    "pulp",
)


def check_module(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except ImportError as exc:
        return {"name": name, "importable": False, "error": f"find_spec ImportError: {exc}"}
    except ValueError as exc:
        return {"name": name, "importable": False, "error": f"find_spec ValueError: {exc}"}
    if spec is None:
        return {"name": name, "importable": False, "error": "spec_not_found"}
    try:
        mod = importlib.import_module(name)
    except Exception as exc:                     # noqa: BLE001 (report any import err)
        return {"name": name, "importable": False,
                "error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": name,
        "importable": True,
        "version": getattr(mod, "__version__", None),
        "file": getattr(mod, "__file__", None),
    }


def report(modules: list[str] | None = None) -> dict[str, Any]:
    to_check = list(modules or PROJECT_MODULES)
    results = [check_module(m) for m in to_check]
    return {
        "checked": to_check,
        "importable": [r["name"] for r in results if r["importable"]],
        "missing": [r["name"] for r in results if not r["importable"]],
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline import check")
    default_comp = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output",
        default=str(default_comp / "artifacts" / "m39_offline_dependency_check.json"),
    )
    ns = parser.parse_args(argv)
    r = report()
    Path(ns.output).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.output).write_text(json.dumps(r, indent=2, sort_keys=True))
    print(f"WROTE {ns.output}  importable={len(r['importable'])}/{len(r['checked'])}  "
          f"missing={r['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
