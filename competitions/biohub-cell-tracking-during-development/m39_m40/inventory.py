"""M39 Step-0 repository inventory.

Scans the competition directory for milestone files, notebooks, runners,
reports, manifests, DB/tracker files, saved outputs, and reference
artifacts. Emits ``artifacts/m39_repo_inventory.json``.

This module NEVER runs GPU code, NEVER imports torch, and NEVER hits the
Kaggle mount. It is safe to import in any environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


COMP_DIR_DEFAULT = (
    Path(__file__).resolve().parents[1]
)
"""Default competition dir = parent of m39_m40/."""

INVENTORY_PATTERNS: dict[str, list[str]] = {
    "one_cell_txt": ["M*_*.txt", "KAGGLE_*.txt"],
    "runners_py": ["milestone*_runner*.py", "M*_*RUNNER*.py", "M*_SUBMIT*.py"],
    "kaggle_cells_py": ["kaggle_cell_*.py"],
    "notebooks": ["*.ipynb"],
    "reports_md": ["*REPORT*.md", "*IMPLEMENTATION*.md"],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_first_mb(path: Path) -> str:
    """SHA of the first MB — used for very large files whose full hash would
    dominate inventory time. Marked as such in the record."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
    return h.hexdigest()


def _record_file(path: Path, *, full_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    if stat.st_size > (16 << 20) and not full_hash:
        digest = _sha256_first_mb(path)
        hash_scope = "first_mb"
    else:
        digest = _sha256(path)
        hash_scope = "full"
    return {
        "path": str(path.relative_to(COMP_DIR_DEFAULT.parent.parent)),
        "size_bytes": stat.st_size,
        "sha256": digest,
        "sha_scope": hash_scope,
    }


def _sorted_glob(base: Path, patterns: list[str]) -> list[Path]:
    hits: set[Path] = set()
    for pat in patterns:
        hits.update(p for p in base.glob(pat) if p.is_file())
    return sorted(hits)


def _git_facts(repo_root: Path) -> dict[str, Any]:
    def _cmd(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "repo_root": str(repo_root),
        "head_sha": _cmd(["git", "rev-parse", "HEAD"]),
        "branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_cmd(["git", "status", "--porcelain"])),
    }


def _env_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "has_torch": False,
        "torch_version": None,
        "cuda_available": False,
        "kaggle_mount_present": Path(
            "/kaggle/input/competitions/biohub-cell-tracking-during-development"
        ).is_dir(),
    }
    try:
        import torch  # type: ignore  # noqa: F401
        facts["has_torch"] = True
        facts["torch_version"] = getattr(torch, "__version__", None)  # type: ignore
        try:
            facts["cuda_available"] = bool(torch.cuda.is_available())  # type: ignore
        except Exception:      # noqa: BLE001
            facts["cuda_available"] = False
    except ImportError:
        pass
    return facts


def _existing_paths(paths: list[Path]) -> list[dict[str, Any]]:
    return [_record_file(p, full_hash=False) for p in paths if p.is_file()]


def _archive_m19c_facts(comp_dir: Path) -> dict[str, Any]:
    archive = comp_dir / "archive" / "m19c"
    manifest = archive / "MANIFEST.sha256"
    files: list[dict[str, Any]] = []
    manifest_ok = False
    if manifest.exists():
        try:
            wanted: dict[str, str] = {}
            for line in manifest.read_text().splitlines():
                sha, path = line.split("  ", 1)
                wanted[path] = sha
            all_ok = True
            for name, want in wanted.items():
                p = archive / name
                if not p.is_file():
                    all_ok = False
                    files.append({"name": name, "sha256_expected": want, "present": False})
                    continue
                got = _sha256(p)
                files.append({
                    "name": name,
                    "sha256_expected": want,
                    "sha256_got": got,
                    "present": True,
                    "match": got == want,
                })
                if got != want:
                    all_ok = False
            manifest_ok = all_ok
        except (OSError, ValueError):
            manifest_ok = False
    return {
        "archive_path": str(archive),
        "manifest_present": manifest.exists(),
        "manifest_ok": manifest_ok,
        "files": files,
    }


def build_inventory(comp_dir: Path | None = None) -> dict[str, Any]:
    comp = (comp_dir or COMP_DIR_DEFAULT).resolve()
    repo_root = comp.parents[1]

    milestone_files = {
        kind: _existing_paths(_sorted_glob(comp, patterns))
        for kind, patterns in INVENTORY_PATTERNS.items()
    }

    # Reference / support-pack search targets (existence only)
    reference_targets = [
        comp / ".local_reference" / "biohub-competition-solution.ipynb",
        comp / "reference" / "biohub-competition-solution.ipynb",
        comp / "reference" / "official_metric",
        comp / "reference" / "tracking_cellmot",
        comp / "reference" / "hoct",
        comp / "reference" / "hoct_weights",
        comp / "artifacts",
    ]
    reference = [
        {"path": str(p.relative_to(repo_root)), "exists": p.exists()}
        for p in reference_targets
    ]

    # M19-C fallback validation
    m19c_archive = _archive_m19c_facts(comp)

    # M38 latest submission runner and its status file
    m38 = {
        "runner_py": _record_file(comp / "M38_HIDDEN_RESILIENT_REFERENCE_0902_SUBMIT.py")
        if (comp / "M38_HIDDEN_RESILIENT_REFERENCE_0902_SUBMIT.py").exists() else None,
        "runner_txt": _record_file(comp / "M38_HIDDEN_RESILIENT_REFERENCE_0902_SUBMIT.txt")
        if (comp / "M38_HIDDEN_RESILIENT_REFERENCE_0902_SUBMIT.txt").exists() else None,
    }

    return {
        "milestone": "M39",
        "generated_by": "m39_m40.inventory.build_inventory",
        "competition_dir": str(comp),
        "git": _git_facts(repo_root),
        "env": _env_facts(),
        "counts": {kind: len(files) for kind, files in milestone_files.items()},
        "milestone_files": milestone_files,
        "reference_and_support": reference,
        "m19c_archive": m19c_archive,
        "m38_runner": m38,
    }


def write_inventory(
    inventory: dict[str, Any],
    output_path: Path | str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M39 repository inventory")
    parser.add_argument(
        "--comp-dir",
        default=str(COMP_DIR_DEFAULT),
        help="Competition directory (default: parent of m39_m40/).",
    )
    parser.add_argument(
        "--output",
        default=str(COMP_DIR_DEFAULT / "artifacts" / "m39_repo_inventory.json"),
        help="Output JSON path.",
    )
    ns = parser.parse_args(argv)
    inv = build_inventory(Path(ns.comp_dir))
    written = write_inventory(inv, ns.output)
    print(f"WROTE {written}  files_counted={sum(inv['counts'].values())}  "
          f"env.has_torch={inv['env']['has_torch']}  "
          f"env.cuda={inv['env']['cuda_available']}  "
          f"env.kaggle_mount={inv['env']['kaggle_mount_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
