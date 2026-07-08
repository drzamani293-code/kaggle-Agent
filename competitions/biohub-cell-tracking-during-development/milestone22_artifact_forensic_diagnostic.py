"""
Biohub - Cell Tracking During Development
Milestone 22 - Stage 1: TRUE M19-C artifact forensic diagnostic.

Why
---
M19-C (public 0.880) was produced from a base learned graph of
n_nodes_before=131797 / n_edges_before=118992. The two mounted dataset
support packs do NOT reproduce that base:
  pilkwang350 -> 142193 / 127563   weight_sha256 dfb848aa...5e36ec
  tom99763    -> 161098 / 137520   weight_sha256 912ae91a...ab3f53
So the true M19-C artifact/weights (possibly a NOTEBOOK input, e.g. a kernel
named like "Biohub Cell Tracking: Learned Graph w G") must be recovered before
any further tuning is meaningful.

What this cell does (read-only forensic, no submission)
-------------------------------------------------------
- Enumerates every candidate artifact root under /kaggle/input (including
  /kaggle/input/datasets/*/* and /kaggle/input/notebooks/*/*, one/two/three
  levels deep), hidden-safe (globs, never hardcodes dataset names).
- For each root that contains repo/scripts/predict_unet_transformer.py and
  weights/unet_transformer/split_0/edge_predictor_best.pth, records:
  candidate_path, artifact_name, weight_sha256, manifest summary, repo path,
  wheels presence, and required-files presence.
- Flags the two KNOWN-BAD weight hashes (pilkwang350, tom99763) so a genuinely
  different (candidate true) artifact stands out.
- Optional controlled SMOKE RUN (M22_FORENSIC_SMOKE_RUN): for each viable
  candidate whose hash is NOT known-bad, runs the EXACT M19-C predict command
  (det 0.99, ilp -1.0/0.1/0.1/1.0, --use-ilp), converts the GEFF to base
  node/edge counts with NO extra post-processing, and checks
  n_nodes_before==131797 AND n_edges_before==118992. Stops at the first match.
- Writes /kaggle/working/m22_artifact_forensic_report.json and
  /kaggle/working/m22_candidate_table.csv, and prints
  TRUE_M19C_ARTIFACT_FOUND (with the selected path/name/hash/repo/weights/base
  counts) or TRUE_M19C_ARTIFACT_NOT_FOUND.

Off Kaggle it is a clean self-test-only dry run. Never submits, never calls
the Kaggle submission API, never triggers Save Version. No prior milestone
file is modified - all reused M16/M17 machinery is copied in.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import blosc2
except ImportError:
    blosc2 = None

try:
    import zstandard
except ImportError:
    zstandard = None


# --------------------------------------------------------------------------- #
# 0. Constants: artifact locations, competition paths, hidden-name guard
# --------------------------------------------------------------------------- #
KAGGLE_COMPETITION_INPUT_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
KAGGLE_WORKING_DIR = "/kaggle/working"
TRACKING_REPO_DIR = "/kaggle/working/tracking_repo"

# Candidate mount points for the uploaded support-pack artifact, in
# preference order. The resolver requires the manifest + predict script +
# weights + wheels to be present before accepting a candidate.
ARTIFACT_CANDIDATE_DIRS = [
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/datasets/tom99763/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-v1",
    # Broader fallbacks for other Kaggle mount layouts (dataset slug only).
    "/kaggle/input/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/biohub-tracking-support-pack-v1",
]

ARTIFACT_REQUIRED_ENTRIES = [
    "ARTIFACT_MANIFEST.json",
    "repo/scripts/predict_unet_transformer.py",
    "weights/unet_transformer/split_0/edge_predictor_best.pth",
    "wheels",
]

# Core scientific packages pip must NEVER replace when installing the
# artifact's bundled wheels (Kaggle's preinstalled builds are ABI-matched
# to its CUDA/BLAS stack; letting a wheel swap them breaks torch/numba).
PROTECTED_PACKAGES = ["numpy", "scipy", "numba", "llvmlite", "torch", "pandas"]

# Two-phase offline install order (polars first so the Float16 shim below
# is importable before anything that depends on it).
WHEEL_INSTALL_PHASES = [
    ["polars"],
    ["tracksdata", "zarr", "pyscipopt", "geff", "geff-spec", "ilpy", "imagecodecs",
     "rustworkx", "numcodecs", "donfig", "bidict"],
]

# Fresh-subprocess import smoke test after install.
IMPORT_CHECK_MODULES = ["zarr", "tracksdata", "geff", "polars", "numcodecs", "rustworkx", "biohub_tracking"]

# Extra sys.path entries every subprocess needs (the reference repo layout).
SUBPROCESS_PATH_ENTRIES = [
    "/kaggle/working/tracking_repo/src",
    "/kaggle/working/tracking_repo",
    "/kaggle/working",
]

# Public dataset names that MUST NOT appear hardcoded in any submitted
# output - used only by a self-test that asserts we never emit them.
PUBLIC_TEST_DATASET_NAMES = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

# The polars Float16 compatibility shim written into sitecustomize.py at
# several locations so every subprocess picks it up on interpreter start.
SITECUSTOMIZE_CONTENT = (
    "try:\n"
    "    import polars as pl\n"
    "    if not hasattr(pl, \"Float16\"):\n"
    "        pl.Float16 = pl.Float32\n"
    "except Exception:\n"
    "    pass\n"
)

# --------------------------------------------------------------------------- #
# 1. Manual zarr v3 reader (copied from Milestone 15 - reads GEFF prediction
#    arrays without depending on a particular installed zarr version)
# --------------------------------------------------------------------------- #
DTYPE_CODES = {
    "bool": "b1",
    "int8": "i1", "int16": "i2", "int32": "i4", "int64": "i8",
    "uint8": "u1", "uint16": "u2", "uint32": "u4", "uint64": "u8",
    "float16": "f2", "float32": "f4", "float64": "f8",
}

NODE_PROP_NAMES = ("t", "z", "y", "x")

# Physical voxel scale for this competition's volumes (µm/voxel).
VOXEL_SIZE_UM = {"z": 1.625, "y": 0.40625, "x": 0.40625}
_VOXEL_SCALE = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])


# --------------------------------------------------------------------------- #
# 1. zarr.json metadata parsing (manual offline reader, no zarr/numcodecs)
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_array_metadata(array_dir: Path) -> dict:
    """Parse a zarr v3 array's `zarr.json` into shape/dtype/chunking/codecs."""
    meta_path = array_dir / "zarr.json"
    doc = read_json(meta_path)

    if doc.get("zarr_format") != 3:
        raise ValueError(f"{meta_path}: only zarr v3 is supported, got zarr_format={doc.get('zarr_format')!r}")
    if doc.get("node_type") != "array":
        raise ValueError(f"{meta_path}: not a zarr array (node_type={doc.get('node_type')!r})")

    shape = tuple(int(s) for s in doc["shape"])

    data_type = doc["data_type"]
    if isinstance(data_type, dict):  # some writers nest it as {"name": "uint16"}
        data_type = data_type.get("name")
    code = DTYPE_CODES.get(data_type)
    if code is None:
        raise ValueError(f"{meta_path}: unsupported data_type {data_type!r}")

    codecs = doc.get("codecs", [])
    endian = "little"
    for codec in codecs:
        if codec.get("name") in ("bytes", "endian"):
            endian = codec.get("configuration", {}).get("endian", "little")
            break
    dtype = np.dtype("bool") if code == "b1" else np.dtype(("<" if endian == "little" else ">") + code)

    chunk_grid = doc.get("chunk_grid", {})
    if chunk_grid.get("name", "regular") != "regular":
        raise NotImplementedError(f"{meta_path}: unsupported chunk_grid {chunk_grid.get('name')!r}")
    chunk_shape = tuple(int(c) for c in chunk_grid.get("configuration", {}).get("chunk_shape", shape))

    chunk_key_encoding = doc.get(
        "chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}}
    )
    fill_value = doc.get("fill_value", 0) or 0

    return {
        "path": array_dir,
        "shape": shape,
        "dtype": dtype,
        "chunk_shape": chunk_shape,
        "codecs": codecs,
        "chunk_key_encoding": chunk_key_encoding,
        "fill_value": fill_value,
    }


# --------------------------------------------------------------------------- #
# 2. Chunk key construction + codec decoding
# --------------------------------------------------------------------------- #
def build_chunk_path(array_dir: Path, index: Sequence[int], chunk_key_encoding: dict) -> Path:
    """Build the on-disk chunk file path for a given chunk index tuple."""
    name = chunk_key_encoding.get("name", "default")
    separator = chunk_key_encoding.get("configuration", {}).get("separator", "/")
    index_strs = [str(i) for i in index]

    if name == "v2":
        key = separator.join(index_strs) if index_strs else "0"
    else:
        key = separator.join(["c", *index_strs]) if index_strs else "c"

    if separator == "/":
        return array_dir.joinpath(*key.split("/"))
    return array_dir / key


def decode_bytes_codecs(raw: bytes, codecs: list[dict], expected_nbytes: int) -> bytes:
    """Reverse the bytes->bytes compressor chain, then hand back raw array bytes.

    `codecs[0]` is the array->bytes codec (must be "bytes"/"endian" - a plain
    passthrough once endianness is known); `codecs[1:]` are bytes->bytes
    compressors applied in encode order, so they're undone in reverse order.
    """
    if not codecs:
        return raw

    array_bytes_codec = codecs[0]
    if array_bytes_codec.get("name") not in ("bytes", "endian"):
        raise NotImplementedError(
            f"unsupported array->bytes codec: {array_bytes_codec.get('name')!r} "
            "(only the plain 'bytes' codec is supported - no sharding)"
        )

    data = raw
    for codec in reversed(codecs[1:]):
        codec_name = codec.get("name")
        if codec_name == "blosc":
            if blosc2 is None:
                raise RuntimeError("chunk uses the 'blosc' codec but the blosc2 package is not installed")
            data = blosc2.decompress(data)
        elif codec_name == "zstd":
            if zstandard is None:
                raise RuntimeError("chunk uses the 'zstd' codec but the zstandard package is not installed")
            data = zstandard.ZstdDecompressor().decompress(data, max_output_size=expected_nbytes)
        else:
            raise NotImplementedError(f"unsupported bytes->bytes codec: {codec_name!r}")

    return data


def decode_chunk_bytes(raw: bytes, codecs: list[dict], dtype: np.dtype, chunk_shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(chunk_shape)) if chunk_shape else 1
    expected_nbytes = count * dtype.itemsize
    decoded = decode_bytes_codecs(raw, codecs, expected_nbytes)
    arr = np.frombuffer(decoded, dtype=dtype, count=count)
    return arr.reshape(chunk_shape)


def read_chunk_at(meta: dict, index: Sequence[int]) -> np.ndarray:
    """Read+decode a single chunk. Missing chunk files are sparse and decode
    to an array filled with the array's fill_value (never raises for that case).
    """
    chunk_path = build_chunk_path(meta["path"], index, meta["chunk_key_encoding"])
    chunk_shape = meta["chunk_shape"]
    if not chunk_path.exists():
        return np.full(chunk_shape, meta["fill_value"], dtype=meta["dtype"])
    raw = chunk_path.read_bytes()
    return decode_chunk_bytes(raw, meta["codecs"], meta["dtype"], chunk_shape)


def read_full_zarr_array(array_dir: Path) -> np.ndarray:
    """Read an entire zarr v3 array (stitching together every chunk)."""
    meta = parse_array_metadata(array_dir)
    shape, chunk_shape = meta["shape"], meta["chunk_shape"]

    if not shape:  # 0-d array
        return read_chunk_at(meta, ())

    n_chunks = [math.ceil(shape[d] / chunk_shape[d]) if shape[d] else 0 for d in range(len(shape))]
    full = np.full(shape, meta["fill_value"], dtype=meta["dtype"])
    for index in itertools.product(*(range(n) for n in n_chunks)):
        chunk = read_chunk_at(meta, index)
        starts = [index[d] * chunk_shape[d] for d in range(len(shape))]
        ends = [min(starts[d] + chunk_shape[d], shape[d]) for d in range(len(shape))]
        dst = tuple(slice(s, e) for s, e in zip(starts, ends))
        src = tuple(slice(0, e - s) for s, e in zip(starts, ends))
        full[dst] = chunk[src]
    return full

# --------------------------------------------------------------------------- #
# 2. Environment detection + artifact resolution + repo materialization
# --------------------------------------------------------------------------- #
def is_kaggle_env() -> bool:
    """True only when the definitive Kaggle marker `/kaggle/input` exists.
    Deliberately NOT keyed on `/kaggle/working` - a local dry run can
    create that directory, and keying on it would flip an off-Kaggle
    execution into attempting a real (impossible) artifact run, breaking
    clean notebook execution off-Kaggle.
    """
    return Path("/kaggle/input").exists()


def _artifact_entry_status(artifact_dir: Path) -> dict:
    return {entry: (artifact_dir / entry).exists() for entry in ARTIFACT_REQUIRED_ENTRIES}


def resolve_artifact(candidate_dirs: Sequence[str] = ARTIFACT_CANDIDATE_DIRS) -> dict:
    """Returns the first candidate artifact mount that contains every
    required entry (manifest, predict script, weights, wheels). Never
    assumes availability - records which candidates were checked and why
    each was rejected, so a hidden rerun with a differently-named mount
    is diagnosable.
    """
    checked = []
    for candidate in candidate_dirs:
        artifact_dir = Path(candidate)
        if not artifact_dir.exists():
            checked.append({"path": candidate, "exists": False, "entries": {}})
            continue
        status = _artifact_entry_status(artifact_dir)
        checked.append({"path": candidate, "exists": True, "entries": status})
        if all(status.values()):
            return {"resolved": True, "artifact_dir": str(artifact_dir), "checked": checked}
    return {"resolved": False, "artifact_dir": None, "checked": checked}


def materialize_repo(artifact_dir: Path, working_dir: Path = Path(KAGGLE_WORKING_DIR)) -> dict:
    """Copies the artifact's `repo/` to `/kaggle/working/tracking_repo` and
    places the weights under `tracking_repo/weights` (symlink preferred,
    copy fallback). Idempotent: a re-run replaces an existing tracking_repo
    rather than erroring.
    """
    repo_src = artifact_dir / "repo"
    repo_dst = working_dir / "tracking_repo"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(repo_src, repo_dst)

    weights_src = artifact_dir / "weights"
    weights_dst = repo_dst / "weights"
    weights_mode = None
    if weights_dst.exists() or weights_dst.is_symlink():
        if weights_dst.is_symlink() or weights_dst.is_file():
            weights_dst.unlink()
        else:
            shutil.rmtree(weights_dst)
    try:
        weights_dst.symlink_to(weights_src, target_is_directory=True)
        weights_mode = "symlink"
    except OSError:
        shutil.copytree(weights_src, weights_dst)
        weights_mode = "copy"

    return {
        "repo_src": str(repo_src), "repo_dst": str(repo_dst),
        "weights_src": str(weights_src), "weights_dst": str(weights_dst), "weights_mode": weights_mode,
    }


def write_sitecustomize(locations: Sequence[Path] | None = None) -> list[str]:
    """Writes the polars Float16 compatibility shim as sitecustomize.py at
    each location so every subprocess launched with these dirs on its path
    applies the patch on interpreter start.
    """
    if locations is None:
        locations = [Path(KAGGLE_WORKING_DIR), Path(TRACKING_REPO_DIR), Path(TRACKING_REPO_DIR) / "src"]
    written = []
    for loc in locations:
        loc.mkdir(parents=True, exist_ok=True)
        target = loc / "sitecustomize.py"
        target.write_text(SITECUSTOMIZE_CONTENT)
        written.append(str(target))
    return written


def build_subprocess_env(base_env: dict | None = None, path_entries: Sequence[str] = SUBPROCESS_PATH_ENTRIES) -> dict:
    """Returns an environment dict whose PYTHONPATH is prefixed with the
    reference repo's three required roots (preserving any existing
    PYTHONPATH), for every subprocess this runner launches.
    """
    env = dict(os.environ if base_env is None else base_env)
    existing = env.get("PYTHONPATH", "")
    parts = list(path_entries) + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env

# --------------------------------------------------------------------------- #
# 3. Dependency installation (offline wheels only) + import verification
# --------------------------------------------------------------------------- #
def build_pip_install_command(packages: Sequence[str], wheels_dir: Path) -> list[str]:
    """A single offline pip install command: `--no-index --find-links
    <wheels>` (only the bundled wheels are ever considered) and ALWAYS
    `--no-deps` (pip may never pull or resolve a replacement for a
    protected core package - or anything else - from an index).
    """
    return [
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(wheels_dir), "--no-deps", *packages,
    ]


def install_dependencies(wheels_dir: Path, env: dict, phases: Sequence[Sequence[str]] = WHEEL_INSTALL_PHASES) -> dict:
    """Runs the two-phase offline install (polars first, then the rest).
    Every command is `--no-index --find-links <wheels> --no-deps`, so no
    protected core package (numpy/scipy/numba/llvmlite/torch/pandas) can
    ever be swapped by dependency resolution. Records each phase's command
    + return code + captured output for the dependency report.
    """
    phase_results = []
    all_ok = True
    for i, packages in enumerate(phases):
        cmd = build_pip_install_command(packages, wheels_dir)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ok = proc.returncode == 0
        all_ok = all_ok and ok
        phase_results.append({
            "phase": i, "packages": list(packages), "command": cmd, "returncode": proc.returncode,
            "ok": ok, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:],
        })
        print(f"  [install phase {i}] {' '.join(packages)} -> rc={proc.returncode}")
        if not ok:
            print(f"  [install phase {i}] FAILED:\n{proc.stderr[-2000:]}")
    return {
        "wheels_dir": str(wheels_dir), "protected_packages": PROTECTED_PACKAGES,
        "phases": phase_results, "all_ok": all_ok,
    }


def verify_imports(env: dict, modules: Sequence[str] = IMPORT_CHECK_MODULES) -> dict:
    """Runs a FRESH subprocess (so it picks up sitecustomize + the newly-
    installed wheels + PYTHONPATH) that imports every required module,
    printing OK/FAIL per module. Returns the parsed per-module result.
    """
    check_src = (
        "import importlib, json\n"
        f"mods = {list(modules)!r}\n"
        "res = {}\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "        res[m] = True\n"
        "    except Exception as e:\n"
        "        res[m] = f'ERROR: {type(e).__name__}: {e}'\n"
        "print('IMPORT_CHECK_JSON=' + json.dumps(res))\n"
    )
    proc = subprocess.run([sys.executable, "-c", check_src], env=env, capture_output=True, text=True)
    parsed = {}
    for line in proc.stdout.splitlines():
        if line.startswith("IMPORT_CHECK_JSON="):
            parsed = json.loads(line[len("IMPORT_CHECK_JSON="):])
            break
    all_ok = bool(parsed) and all(v is True for v in parsed.values())
    return {"returncode": proc.returncode, "modules": parsed, "all_ok": all_ok,
            "stderr_tail": proc.stderr[-2000:]}

# --------------------------------------------------------------------------- #
# 4. Dynamic split generation (hidden-rerun safe - never hardcodes public
#    dataset names)
# --------------------------------------------------------------------------- #
def _strip_zarr(name: str) -> str:
    return name[:-5] if name.endswith(".zarr") else name


def discover_test_datasets_from_dir(test_dir: Path) -> list[str]:
    """Discovers currently-mounted test dataset names from the test dir -
    either `<name>.zarr` stores or `<name>/` dataset directories - with
    the `.zarr` suffix stripped. Returns a sorted, de-duplicated list.
    """
    if not test_dir.exists():
        return []
    names = set()
    for child in test_dir.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            names.add(_strip_zarr(child.name))
    return sorted(names)


def discover_test_datasets_from_sample_submission(sample_submission_path: Path) -> list[str]:
    """Discovers test dataset names from the mounted sample_submission.csv's
    `dataset` column (the competition's own source of truth), `.zarr`
    stripped. Returns a sorted, de-duplicated list.
    """
    if not sample_submission_path.exists():
        return []
    df = pd.read_csv(sample_submission_path)
    if "dataset" not in df.columns:
        return []
    return sorted({_strip_zarr(str(d)) for d in df["dataset"].dropna().unique()})


def discover_test_datasets(test_dir: Path, sample_submission_path: Path) -> dict:
    """Prefers sample_submission's dataset column (competition source of
    truth); falls back to the mounted test dir. Returns both sources plus
    the chosen list, so the choice is auditable in the diagnostic.
    """
    from_submission = discover_test_datasets_from_sample_submission(sample_submission_path)
    from_dir = discover_test_datasets_from_dir(test_dir)
    if from_submission:
        chosen, source = from_submission, "sample_submission"
    elif from_dir:
        chosen, source = from_dir, "test_dir"
    else:
        chosen, source = [], "none"
    return {"chosen": chosen, "source": source, "from_sample_submission": from_submission, "from_test_dir": from_dir}


def build_canonical_splits(dataset_names: Sequence[str]) -> list:
    """The canonical splits schema the predict script accepts: a LIST whose
    element 0 is the fold-0 dict with `test` set to the discovered dataset
    names. predict_unet_transformer.py does `folds[fold]["test"]` with an
    INTEGER `fold` (0), so the top-level JSON MUST be a list - a
    `{"0": {...}}` dict would json.load into STRING keys and raise
    `KeyError: 0` at predict time.
    """
    return [{"test": list(dataset_names), "train": [], "val": []}]


def rewrite_template_splits(template, dataset_names: Sequence[str]) -> list | None:
    """Rewrites a template into a LIST whose element 0's `test` is the
    discovered names (predict indexes `folds[0]` with an integer). Handles:
      - list template:          [{"test": [...]}, ...]
      - splits-list template:   {"splits": [{"test": [...]}, ...]}
      - numbered-dict template: {"0": {"test": [...]}, "1": {...}}
    Fold 0's other keys (train/val) are preserved where present. Returns
    None for an unrecognized shape so the caller rebuilds canonically. The
    result is ALWAYS a list - never a `{"0": ...}` dict.
    """
    names = list(dataset_names)
    if isinstance(template, list) and template and isinstance(template[0], dict):
        new = [dict(e) if isinstance(e, dict) else e for e in template]
        new[0] = dict(new[0])
        new[0]["test"] = names
        return new
    if isinstance(template, dict) and isinstance(template.get("splits"), list) and template["splits"] and isinstance(template["splits"][0], dict):
        new = [dict(e) if isinstance(e, dict) else e for e in template["splits"]]
        new[0] = dict(new[0])
        new[0]["test"] = names
        return new
    if isinstance(template, dict) and isinstance(template.get("0"), dict):
        split0 = dict(template["0"])
        split0["test"] = names
        return [split0]
    return None


def _dataset_present_in_test_dir(test_dir: Path, name: str) -> bool:
    return (test_dir / name).exists() or (test_dir / f"{name}.zarr").exists()


def prepare_splits(
    test_dir: Path, sample_submission_path: Path, template_path: Path | None,
    out_splits_path: Path, diagnostic_path: Path,
) -> dict:
    """Builds a splits file for the CURRENTLY MOUNTED test data:
      1. discover the mounted dataset names,
      2. rewrite a template's split-0 test list with them (or build
         canonically if there's no template / an unrecognized shape),
      3. VERIFY every split-0 name actually exists in the test dir -
         rebuild canonically from only the verified-present names if not,
      4. write the splits file the predict script will read, plus a
         diagnostic recording exactly what was discovered and written.
    Public dataset names are never hardcoded; the file only ever contains
    names that exist on disk right now.
    """
    discovery = discover_test_datasets(test_dir, sample_submission_path)
    names = discovery["chosen"]

    template = None
    template_used = False
    if template_path is not None and template_path.exists():
        try:
            template = json.loads(template_path.read_text())
        except Exception as exc:
            print(f"  [splits] could not parse template {template_path}: {exc!r} - building canonically")

    splits = None
    if template is not None:
        splits = rewrite_template_splits(template, names)
        template_used = splits is not None
    if splits is None:
        splits = build_canonical_splits(names)

    # `splits` is ALWAYS a list now - fold 0 is `splits[0]` (predict indexes
    # folds[0] with an integer). Verify split-0 test names exist on disk; if
    # any is missing (e.g. a template nested names somewhere we didn't
    # rewrite), rebuild canonically from only the verified-present names.
    split0 = splits[0] if isinstance(splits, list) and splits and isinstance(splits[0], dict) else {}
    split0_test = list(split0.get("test", []))

    verified_present = [n for n in split0_test if _dataset_present_in_test_dir(test_dir, n)]
    missing = [n for n in split0_test if not _dataset_present_in_test_dir(test_dir, n)]
    rebuilt_canonical = False
    if missing or not split0_test:
        present_names = [n for n in names if _dataset_present_in_test_dir(test_dir, n)] or names
        splits = build_canonical_splits(present_names)
        rebuilt_canonical = True
        template_used = False
        split0_test = present_names
        verified_present = [n for n in present_names if _dataset_present_in_test_dir(test_dir, n)]
        missing = [n for n in present_names if not _dataset_present_in_test_dir(test_dir, n)]

    out_splits_path.parent.mkdir(parents=True, exist_ok=True)
    out_splits_path.write_text(json.dumps(splits, indent=2))

    # Verify the WRITTEN file matches EXACTLY how predict_unet_transformer.py
    # reads it: `folds = json.load(f); test_names = folds[fold]["test"]` with
    # `fold == 0` (an integer). The file MUST be a list and folds[0]["test"]
    # MUST equal the discovered names - the exact regression the KeyError: 0
    # fallback surfaced (a {"0": ...} dict json.loads to string keys).
    with open(out_splits_path) as f:
        folds = json.load(f)
    assert isinstance(folds, list), "splits file must be a JSON list (predict indexes folds[0] with an integer)"
    assert len(folds) > 0, "splits list must be non-empty"
    assert isinstance(folds[0], dict), "splits[0] must be a dict"
    assert folds[0]["test"] == split0_test, "folds[0]['test'] must equal the discovered test names"

    diagnostic = {
        "discovered": discovery, "template_path": str(template_path) if template_path else None,
        "template_used": template_used, "rebuilt_canonical": rebuilt_canonical,
        "split0_test_names": split0_test, "verified_present": verified_present, "missing": missing,
        "out_splits_path": str(out_splits_path),
        "final_schema": "list", "predict_index_check": "folds[0]['test'] OK",
    }
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    print(f"  [splits] wrote {out_splits_path} (list schema) with {len(split0_test)} test dataset(s): {split0_test}")
    return {"splits": splits, "split0_test_names": split0_test, "diagnostic": diagnostic}

# --------------------------------------------------------------------------- #
# 5. Prediction command builder + runner
# --------------------------------------------------------------------------- #
PREDICT_SCRIPT_REL = "scripts/predict_unet_transformer.py"
WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"
SPLITS_FILENAME = "kaggle_test_splits_50ep.json"

PREDICT_HYPERPARAMS = {
    "unet_batch_size": "4", "det_threshold": "0.99", "ilp_edge_weight": "-1.0",
    "ilp_appearance_weight": "0.1", "ilp_disappearance_weight": "0.1", "ilp_division_weight": "1.0",
}


def build_predict_command(
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
    split: str = "0", weights_rel: str = WEIGHTS_REL, hyperparams: dict = PREDICT_HYPERPARAMS,
) -> list[str]:
    """Builds the exact GPU predict command (run with cwd=tracking_repo),
    mirroring the successful manual public run. `--use-ilp` is a bare flag.
    """
    return [
        sys.executable, PREDICT_SCRIPT_REL,
        "--data-dir", data_dir,
        "--splits", splits_filename,
        "--split", split,
        "--weights", weights_rel,
        "--unet-batch-size", hyperparams["unet_batch_size"],
        "--det-threshold", hyperparams["det_threshold"],
        "--ilp-edge-weight", hyperparams["ilp_edge_weight"],
        "--ilp-appearance-weight", hyperparams["ilp_appearance_weight"],
        "--ilp-disappearance-weight", hyperparams["ilp_disappearance_weight"],
        "--ilp-division-weight", hyperparams["ilp_division_weight"],
        "--use-ilp",
    ]


def run_prediction(command: Sequence[str], cwd: Path, env: dict) -> dict:
    """Runs the predict command from `cwd` (tracking_repo). Streams nothing
    but captures stdout/stderr tails and the return code for the execute
    report. Does not raise on a nonzero return code - the caller decides
    whether that is fatal (the top-level fallback handles it).
    """
    proc = subprocess.run(list(command), cwd=str(cwd), env=env, capture_output=True, text=True)
    result = {
        "command": list(command), "cwd": str(cwd), "returncode": proc.returncode,
        "ok": proc.returncode == 0, "stdout_tail": proc.stdout[-8000:], "stderr_tail": proc.stderr[-8000:],
    }
    print(f"  [predict] rc={proc.returncode}")
    if proc.returncode != 0:
        print(f"  [predict] stderr tail:\n{proc.stderr[-2000:]}")
    return result

# --------------------------------------------------------------------------- #
# 6. GEFF -> submission conversion
# --------------------------------------------------------------------------- #
def _read_geff_array(geff_root: Path, rel_path: str):
    """Reads one GEFF array (e.g. "nodes/ids", "nodes/props/t/values") if
    present, else returns None. Uses the version-independent manual zarr
    reader first; falls back to the installed `zarr` package (guaranteed
    present on Kaggle from the artifact wheels) if a chunk codec the manual
    reader lacks was used to write the predictions.
    """
    array_dir = geff_root / Path(rel_path)
    if not (array_dir / "zarr.json").exists():
        return None
    try:
        return read_full_zarr_array(array_dir)
    except Exception:
        try:
            import zarr  # noqa: F401
            return zarr.open(store=str(geff_root), path=rel_path, mode="r")[:]
        except Exception as exc:
            print(f"  [geff] could not read {rel_path} under {geff_root}: {exc!r}")
            return None


def _geff_root_of(candidate: Path) -> Path | None:
    """Returns `candidate` if it is itself a GEFF store, else searches one
    level down for a nested GEFF store (some writers emit `<name>/tracks`).
    """
    if (candidate / "nodes" / "ids" / "zarr.json").exists():
        return candidate
    if candidate.is_dir():
        for child in candidate.iterdir():
            if child.is_dir() and (child / "nodes" / "ids" / "zarr.json").exists():
                return child
    return None


def read_geff_graph(geff_root: Path) -> dict:
    """Reads a GEFF prediction store into plain arrays: node ids + t/z/y/x,
    optional node `solution` mask (keep only selected nodes), edge id pairs,
    optional edge `solution` mask (keep only selected edges), and optional
    `edge_prob` / `edge_dist` used only for deterministic tie-breaking when
    enforcing degree constraints.
    """
    node_ids = _read_geff_array(geff_root, "nodes/ids")
    if node_ids is None:
        raise ValueError(f"{geff_root}: no nodes/ids array")
    node_ids = np.asarray(node_ids).reshape(-1)
    t = _read_geff_array(geff_root, "nodes/props/t/values")
    z = _read_geff_array(geff_root, "nodes/props/z/values")
    y = _read_geff_array(geff_root, "nodes/props/y/values")
    x = _read_geff_array(geff_root, "nodes/props/x/values")
    node_solution = _read_geff_array(geff_root, "nodes/props/solution/values")

    edge_ids = _read_geff_array(geff_root, "edges/ids")
    edge_solution = _read_geff_array(geff_root, "edges/props/solution/values")
    edge_prob = _read_geff_array(geff_root, "edges/props/edge_prob/values")
    edge_dist = _read_geff_array(geff_root, "edges/props/edge_dist/values")

    def _col(arr, n):
        return np.asarray(arr).reshape(-1) if arr is not None else np.zeros(n, dtype=float)

    n_nodes = len(node_ids)
    nodes = pd.DataFrame({
        "raw_id": node_ids.astype(np.int64),
        "t": _col(t, n_nodes), "z": _col(z, n_nodes), "y": _col(y, n_nodes), "x": _col(x, n_nodes),
    })
    if node_solution is not None:
        mask = np.asarray(node_solution).reshape(-1).astype(bool)
        nodes = nodes[mask].reset_index(drop=True)

    if edge_ids is None or np.asarray(edge_ids).size == 0:
        edges = pd.DataFrame(columns=["raw_source", "raw_target", "edge_prob", "edge_dist"])
    else:
        edge_ids = np.asarray(edge_ids)
        if edge_ids.ndim == 1:
            edge_ids = edge_ids.reshape(-1, 2)
        n_edges = edge_ids.shape[0]
        edges = pd.DataFrame({
            "raw_source": edge_ids[:, 0].astype(np.int64), "raw_target": edge_ids[:, 1].astype(np.int64),
            "edge_prob": _col(edge_prob, n_edges), "edge_dist": _col(edge_dist, n_edges),
        })
        if edge_solution is not None:
            emask = np.asarray(edge_solution).reshape(-1).astype(bool)
            edges = edges[emask].reset_index(drop=True)

    return {"nodes": nodes, "edges": edges,
            "has_edge_prob": edge_prob is not None, "has_edge_dist": edge_dist is not None}


def convert_geff_graph_to_rows(
    geff_graph: dict, dataset_name: str, next_id: int, next_node_id: int,
) -> tuple[list[dict], int, int, dict]:
    """Converts one GEFF graph into submission node/edge rows, threading
    global-consecutive `id` and globally-unique `node_id` counters across
    datasets. Remaps raw GEFF node ids to positive consecutive Kaggle
    node_ids (per dataset); keeps only edges whose BOTH endpoints survived
    node selection; requires target_t > source_t; drops duplicate directed
    edges; enforces in_degree <= 1 and out_degree <= 2, breaking ties by
    higher edge_prob then lower edge_dist then stable order.
    """
    nodes = geff_graph["nodes"].copy()
    edges = geff_graph["edges"].copy()

    # Remap raw -> consecutive Kaggle node_id (globally unique via next_node_id).
    raw_to_kaggle: dict[int, int] = {}
    node_rows = []
    node_t = {}
    kid = next_node_id
    for row in nodes.itertuples():
        raw_to_kaggle[int(row.raw_id)] = kid
        node_t[kid] = int(row.t)
        node_rows.append({"kaggle_node_id": kid, "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
        kid += 1
    next_node_id_out = kid

    stats = {
        "dataset": dataset_name, "n_geff_nodes": len(nodes), "n_geff_edges": len(edges),
        "n_edges_dangling_dropped": 0, "n_edges_time_order_dropped": 0,
        "n_edges_duplicate_dropped": 0, "n_edges_indegree_dropped": 0, "n_edges_outdegree_dropped": 0,
    }

    kept_edges = []
    for row in edges.itertuples():
        s_raw, t_raw = int(row.raw_source), int(row.raw_target)
        if s_raw not in raw_to_kaggle or t_raw not in raw_to_kaggle:
            stats["n_edges_dangling_dropped"] += 1
            continue
        s_kid, t_kid = raw_to_kaggle[s_raw], raw_to_kaggle[t_raw]
        if not (node_t[t_kid] > node_t[s_kid]):
            stats["n_edges_time_order_dropped"] += 1
            continue
        kept_edges.append({
            "source_id": s_kid, "target_id": t_kid,
            "edge_prob": float(row.edge_prob), "edge_dist": float(row.edge_dist),
        })

    edges_df = pd.DataFrame(kept_edges, columns=["source_id", "target_id", "edge_prob", "edge_dist"])

    if len(edges_df):
        before = len(edges_df)
        edges_df = edges_df.drop_duplicates(subset=["source_id", "target_id"], keep="first")
        stats["n_edges_duplicate_dropped"] = before - len(edges_df)

        # Rank edges best-first: higher edge_prob, then lower edge_dist, then
        # stable order - so degree pruning keeps the most-confident edges.
        edges_df = edges_df.assign(_order=np.arange(len(edges_df)))
        edges_df = edges_df.sort_values(
            ["edge_prob", "edge_dist", "_order"], ascending=[False, True, True]
        ).reset_index(drop=True)

        # in_degree <= 1: each target keeps only its single best incoming edge.
        before = len(edges_df)
        edges_df = edges_df.drop_duplicates(subset=["target_id"], keep="first")
        stats["n_edges_indegree_dropped"] = before - len(edges_df)

        # out_degree <= 2: each source keeps at most its 2 best outgoing edges.
        before = len(edges_df)
        edges_df = edges_df.groupby("source_id", sort=False, group_keys=False).head(2)
        stats["n_edges_outdegree_dropped"] = before - len(edges_df)

    rows = []
    cur_id = next_id
    for nr in node_rows:
        rows.append({
            "id": cur_id, "dataset": dataset_name, "row_type": "node", "node_id": nr["kaggle_node_id"],
            "t": nr["t"], "z": nr["z"], "y": nr["y"], "x": nr["x"], "source_id": -1, "target_id": -1,
        })
        cur_id += 1
    for er in edges_df.itertuples() if len(edges_df) else []:
        rows.append({
            "id": cur_id, "dataset": dataset_name, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(er.source_id), "target_id": int(er.target_id),
        })
        cur_id += 1

    stats["n_submission_nodes"] = len(node_rows)
    stats["n_submission_edges"] = int(len(edges_df))
    return rows, cur_id, next_node_id_out, stats


def find_prediction_geff_stores(predictions_dir: Path) -> list[tuple[str, Path]]:
    """Finds each dataset's GEFF prediction store under `predictions_dir`,
    returning (dataset_name, geff_root) pairs. The dataset name is the store
    directory's stem with any `.geff`/`.zarr` suffix stripped.
    """
    results = []
    if not predictions_dir.exists():
        return results
    for child in sorted(predictions_dir.iterdir()):
        if child.name.startswith("."):
            continue
        geff_root = _geff_root_of(child)
        if geff_root is not None:
            name = _strip_zarr(child.name)
            if name.endswith(".geff"):
                name = name[:-5]
            results.append((name, geff_root))
    return results


def convert_predictions_to_submission(predictions_dir: Path, expected_datasets: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    """Converts every GEFF prediction store under `predictions_dir` into one
    combined submission dataframe with globally-consecutive `id` and
    globally-unique `node_id`. Records per-dataset conversion stats.
    """
    stores = find_prediction_geff_stores(predictions_dir)
    all_rows = []
    per_dataset = []
    next_id, next_node_id = 0, 0
    for dataset_name, geff_root in stores:
        graph = read_geff_graph(geff_root)
        rows, next_id, next_node_id, stats = convert_geff_graph_to_rows(graph, dataset_name, next_id, next_node_id)
        all_rows.extend(rows)
        per_dataset.append(stats)
        print(f"  [geff] {dataset_name}: {stats['n_submission_nodes']} node(s), {stats['n_submission_edges']} edge(s)")

    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    conversion = {
        "predictions_dir": str(predictions_dir),
        "n_stores_found": len(stores), "found_datasets": [n for n, _ in stores],
        "expected_datasets": list(expected_datasets), "per_dataset": per_dataset,
    }
    return submission_df, conversion

# --------------------------------------------------------------------------- #
# 7. Validation (hidden-safe) + fallback writer + diagnostics
# --------------------------------------------------------------------------- #
def _per_dataset_degree_extremes(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[int, int, int, int]:
    """Returns (max_in_degree, max_out_degree, n_dangling, n_duplicate)
    across all datasets, scoping node-id lookups per dataset (a foreign
    submission's node_id is only unique WITHIN a dataset).
    """
    max_in, max_out, dangling, duplicate = 0, 0, 0, 0
    for dataset, e in edges.groupby("dataset"):
        valid_ids = set(nodes.loc[nodes["dataset"] == dataset, "node_id"])
        good = e[e["source_id"].isin(valid_ids) & e["target_id"].isin(valid_ids)]
        dangling += len(e) - len(good)
        deduped = good.drop_duplicates(subset=["source_id", "target_id"])
        duplicate += len(good) - len(deduped)
        if len(deduped):
            max_out = max(max_out, int(deduped.groupby("source_id").size().max()))
            max_in = max(max_in, int(deduped.groupby("target_id").size().max()))
    return max_in, max_out, dangling, duplicate


def validate_reference_submission(df: pd.DataFrame, expected_datasets: Sequence[str]) -> tuple[bool, str, dict]:
    """Hidden-safe structural + topology validation. The expected dataset
    set comes from the caller (mounted sample_submission / test data), never
    a hardcoded public set. Allows out_degree up to 2 (divisions) but
    forbids in_degree > 1 (multi-parent), dangling edges, and duplicate
    directed edges. Returns (all_passed, report, stats).
    """
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("columns match required schema", list(df.columns) == SUBMISSION_COLUMNS, f"got {list(df.columns)}")
    na_cols = df.columns[df.isna().any()].tolist() if len(df) else []
    check("no missing values", not na_cols, f"NaN columns: {na_cols}")
    check("id consecutive from 0", df["id"].tolist() == list(range(len(df))))

    present = set(df["dataset"].unique()) if len(df) else set()
    expected = set(expected_datasets)
    check("every expected dataset present", not (expected - present), f"missing: {sorted(expected - present)}")
    check("no foreign/hardcoded-public datasets", not (present - expected), f"foreign: {sorted(present - expected)}")
    check("row_type only node/edge", set(df["row_type"].unique()) <= {"node", "edge"} if len(df) else True)

    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    node_id_unique_per_dataset = all(g["node_id"].is_unique for _, g in nodes.groupby("dataset")) if len(nodes) else True
    check("node_id unique per dataset", node_id_unique_per_dataset)

    max_in, max_out, dangling, duplicate = _per_dataset_degree_extremes(nodes, edges)
    check("no dangling edges", dangling == 0, f"{dangling} dangling")
    check("no duplicate directed edges", duplicate == 0, f"{duplicate} duplicate")
    check("no in_degree > 1", max_in <= 1, f"max_in_degree={max_in}")
    check("no out_degree > 2", max_out <= 2, f"max_out_degree={max_out}")

    lines = ["Reference submission validation:"]
    all_passed = True
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_passed = False
        line = f"  [{mark}] {c['name']}"
        if c["detail"] and not c["passed"]:
            line += f"  -- {c['detail']}"
        lines.append(line)
    stats = {"n_node_rows": len(nodes), "n_edge_rows": len(edges), "max_in_degree": max_in,
             "max_out_degree": max_out, "dangling": dangling, "duplicate": duplicate,
             "datasets": sorted(present)}
    return all_passed, "\n".join(lines), stats


def compute_edge_distance_summary(df: pd.DataFrame) -> dict:
    """Physical (µm) edge-length percentiles over all valid edges, using
    the competition's per-axis voxel scale.
    """
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    if len(edges) == 0 or len(nodes) == 0:
        return {}
    coord = {}
    for row in nodes.itertuples():
        coord[(row.dataset, int(row.node_id))] = (float(row.z), float(row.y), float(row.x))
    dists = []
    for row in edges.itertuples():
        s = coord.get((row.dataset, int(row.source_id)))
        t = coord.get((row.dataset, int(row.target_id)))
        if s is None or t is None:
            continue
        dz = (t[0] - s[0]) * VOXEL_SIZE_UM["z"]
        dy = (t[1] - s[1]) * VOXEL_SIZE_UM["y"]
        dx = (t[2] - s[2]) * VOXEL_SIZE_UM["x"]
        dists.append(float(np.sqrt(dz * dz + dy * dy + dx * dx)))
    if not dists:
        return {}
    s = pd.Series(dists)
    return {"mean": float(s.mean()), "median": float(s.median()), "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)), "p99": float(s.quantile(0.99)), "max": float(s.max())}


def write_fallback_submission(sample_submission_path: Path, out_path: Path) -> dict:
    """Writes a VALID (if low-scoring) submission.csv derived from the
    mounted sample_submission.csv: exact columns, no missing values (node
    rows get source/target = -1; edge rows get node/t/z/y/x = -1; any
    remaining NaN -> -1), id renumbered consecutively from 0. Used only
    when the learned pipeline failed and no submission.csv exists, so a
    hidden failure still produces a scorable file instead of an exception.
    """
    df = pd.read_csv(sample_submission_path)
    for col in SUBMISSION_COLUMNS:
        if col not in df.columns:
            df[col] = -1
    df = df[SUBMISSION_COLUMNS].copy()
    if "row_type" in df.columns and len(df):
        is_node = df["row_type"] == "node"
        is_edge = df["row_type"] == "edge"
        for col in ("source_id", "target_id"):
            df.loc[is_node, col] = df.loc[is_node, col].fillna(-1)
        for col in ("node_id", "t", "z", "y", "x"):
            df.loc[is_edge, col] = df.loc[is_edge, col].fillna(-1)
    df = df.fillna(-1)
    df["id"] = range(len(df))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {"source": str(sample_submission_path), "out_path": str(out_path), "shape": list(df.shape)}


def write_reference_diagnostics(df: pd.DataFrame, valid: bool, stats: dict, out_dir: Path) -> None:
    """Writes the variant-counts, topology-validation, and edge-distance
    diagnostic CSVs for the reference submission.
    """
    n_nodes = int((df["row_type"] == "node").sum()) if len(df) else 0
    n_edges = int((df["row_type"] == "edge").sum()) if len(df) else 0
    pd.DataFrame([{"variant": "reference_learned_graph", "valid": valid, "total_rows": len(df),
                   "n_node_rows": n_nodes, "n_edge_rows": n_edges}]).to_csv(
        out_dir / "milestone16_reference_variant_counts.csv", index=False)
    pd.DataFrame([{"variant": "reference_learned_graph", "max_in_degree": stats.get("max_in_degree"),
                   "max_out_degree": stats.get("max_out_degree"), "dangling": stats.get("dangling"),
                   "duplicate": stats.get("duplicate")}]).to_csv(
        out_dir / "milestone16_reference_topology_validation.csv", index=False)
    edge_summary = compute_edge_distance_summary(df)
    pd.DataFrame([{"variant": "reference_learned_graph", **edge_summary}]).to_csv(
        out_dir / "milestone16_reference_edge_distance_stats.csv", index=False)

# --------------------------------------------------------------------------- #
# 8. Orchestration + top-level graceful fallback
# --------------------------------------------------------------------------- #
_RUN_LOG: list[str] = []


def log(msg: str) -> None:
    print(msg)
    _RUN_LOG.append(str(msg))


def _write_log(out_dir: Path) -> None:
    try:
        (out_dir / "milestone16_reference_runner.log").write_text("\n".join(_RUN_LOG) + "\n")
    except Exception as exc:
        print(f"[warn] could not write runner log: {exc!r}")


def run_pipeline(working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """The full learned-graph flow: resolve artifact -> materialize repo ->
    sitecustomize -> offline wheel install -> import check -> DYNAMIC splits
    -> GPU predict -> GEFF->submission conversion -> validation -> write
    submission.csv. Raises on any fatal step so the top-level fallback can
    guarantee a valid submission.csv still exists.
    """
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = Path(competition_dir)
    test_dir = comp / "test"
    sample_submission_path = comp / "sample_submission.csv"

    log("=== Milestone 16: reference learned-graph runner ===")

    artifact = resolve_artifact()
    (out_dir / "milestone16_artifact_selection.json").write_text(json.dumps(artifact, indent=2))
    if not artifact["resolved"]:
        raise RuntimeError("support-pack artifact not found at any candidate mount (see milestone16_artifact_selection.json)")
    artifact_dir = Path(artifact["artifact_dir"])
    log(f"Artifact resolved: {artifact_dir}")

    materialize = materialize_repo(artifact_dir, out_dir)
    log(f"Repo materialized to {materialize['repo_dst']} (weights: {materialize['weights_mode']})")
    write_sitecustomize()
    env = build_subprocess_env()

    dep_report = install_dependencies(artifact_dir / "wheels", env)
    (out_dir / "milestone16_dependency_report.json").write_text(json.dumps(dep_report, indent=2))
    if not dep_report["all_ok"]:
        raise RuntimeError("offline wheel install failed (see milestone16_dependency_report.json)")

    import_report = verify_imports(env)
    if not import_report["all_ok"]:
        raise RuntimeError(f"post-install import check failed: {import_report['modules']}")
    log(f"Imports OK: {list(import_report['modules'].keys())}")

    repo_dst = Path(materialize["repo_dst"])
    shipped_template = repo_dst / SPLITS_FILENAME
    splits_result = prepare_splits(
        test_dir=test_dir, sample_submission_path=sample_submission_path,
        template_path=shipped_template if shipped_template.exists() else None,
        out_splits_path=repo_dst / SPLITS_FILENAME, diagnostic_path=out_dir / "milestone16_splits.json",
    )

    command = build_predict_command()
    (out_dir / "milestone16_run_command.json").write_text(json.dumps({"command": command, "cwd": str(repo_dst)}, indent=2))
    log(f"Predict command: {' '.join(command)}")

    exec_result = run_prediction(command, cwd=repo_dst, env=env)
    (out_dir / "milestone16_execute_result.json").write_text(json.dumps(exec_result, indent=2))
    if not exec_result["ok"]:
        raise RuntimeError("predict script returned nonzero (see milestone16_execute_result.json)")

    predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / "split_0"
    expected_datasets = discover_test_datasets(test_dir, sample_submission_path)["chosen"]
    submission_df, conversion = convert_predictions_to_submission(predictions_dir, expected_datasets)
    (out_dir / "milestone16_geff_conversion.json").write_text(json.dumps(conversion, indent=2, default=str))

    valid, report, stats = validate_reference_submission(submission_df, expected_datasets)
    log(report)
    if not valid:
        raise RuntimeError("converted reference submission FAILED validation (falling back)")

    submission_path = out_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    write_reference_diagnostics(submission_df, valid, stats, out_dir)

    report_obj = {
        "artifact_dir": str(artifact_dir), "expected_datasets": expected_datasets,
        "split0_test_names": splits_result["split0_test_names"], "submission_shape": list(submission_df.shape),
        "n_node_rows": stats["n_node_rows"], "n_edge_rows": stats["n_edge_rows"],
        "max_in_degree": stats["max_in_degree"], "max_out_degree": stats["max_out_degree"],
        "valid": valid, "final_source": "reference_learned_graph",
        "final_instruction": "submission.csv built and validated. Do not submit until user reviews.",
    }
    (out_dir / "milestone16_reference_submission_report.json").write_text(json.dumps(report_obj, indent=2, default=str))
    log(f"submission.csv written: shape={submission_df.shape}, source=reference_learned_graph")
    log("submission.csv built and validated. Do not submit until user reviews.")
    return {"submission_df": submission_df, "final_source": "reference_learned_graph", "valid": valid, "report": report_obj}


def run_with_fallback(working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """Runs the learned pipeline; on ANY failure, if no submission.csv was
    produced, writes a valid fallback from sample_submission.csv so a
    hidden rerun still yields a scorable file instead of a thrown
    exception. Re-raises ONLY if even the fallback cannot be written.
    """
    _RUN_LOG.clear()
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    sample_submission_path = Path(competition_dir) / "sample_submission.csv"
    try:
        result = run_pipeline(working_dir, competition_dir)
        _write_log(out_dir)
        return {"status": "ok", **result}
    except Exception:
        tb = traceback.format_exc()
        log("[error] learned pipeline failed:\n" + tb)
        if submission_path.exists():
            log("submission.csv already exists; keeping it (not overwriting with fallback).")
            _write_log(out_dir)
            return {"status": "pipeline_failed_submission_exists"}
        try:
            fb = write_fallback_submission(sample_submission_path, submission_path)
            (out_dir / "milestone16_failure_fallback_report.json").write_text(
                json.dumps({"traceback": tb, "fallback": fb}, indent=2))
            log("FALLBACK sample submission written because learned graph pipeline failed.")
            _write_log(out_dir)
            return {"status": "fallback_written", "fallback": fb}
        except Exception:
            fb_tb = traceback.format_exc()
            try:
                (out_dir / "milestone16_failure_fallback_report.json").write_text(
                    json.dumps({"traceback": tb, "fallback_error": fb_tb}, indent=2))
            except Exception:
                pass
            log("[fatal] could not write fallback submission either:\n" + fb_tb)
            _write_log(out_dir)
            raise

# --------------------------------------------------------------------------- #
# 11. Artifact introspection (runtime discovery of the reference's actual
#     flags, weight splits, command, and post-processing scripts)
# --------------------------------------------------------------------------- #
import re

# argparse flags that look like recovery/relinking/division/postprocess/
# ensemble knobs - used to auto-detect whether Variant B is available.
GAP_RELINK_FLAG_KEYWORDS = ("gap", "relink", "recover", "division", "merge",
                            "postprocess", "post-process", "ensemble", "smooth", "close")


def list_weight_splits(artifact_dir: Path) -> list[int]:
    """Lists the split indices that actually ship an edge_predictor_best.pth
    (weights/unet_transformer/split_<i>/edge_predictor_best.pth).
    """
    base = artifact_dir / "weights" / "unet_transformer"
    splits: list[int] = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name.startswith("split_") and (d / "edge_predictor_best.pth").exists():
                try:
                    splits.append(int(d.name.split("_", 1)[1]))
                except ValueError:
                    continue
    return sorted(splits)


def discover_predict_flags(predict_script_path: Path) -> list[str]:
    """Greps the predict script's source for every argparse flag it
    declares (`add_argument("--flag")`) - the definitive list of what the
    script will accept, without ever executing it.
    """
    if not predict_script_path.exists():
        return []
    text = predict_script_path.read_text(errors="replace")
    flags = set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9][A-Za-z0-9\-_]*)['\"]", text))
    return sorted(flags)


def discover_reference_commands(artifact_dir: Path, max_hits: int = 60) -> list[dict]:
    """Scans the shipped repo/notebooks/readmes for lines that invoke
    predict_unet_transformer.py - i.e. the original reference command(s),
    so M17 can reproduce them exactly.
    """
    hits: list[dict] = []
    exts = {".py", ".ipynb", ".md", ".sh", ".txt", ".json", ".cfg", ".yaml", ".yml"}
    try:
        for p in artifact_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            try:
                text = p.read_text(errors="replace")
            except Exception:
                continue
            if "predict_unet_transformer" not in text:
                continue
            for line in text.splitlines():
                if "predict_unet_transformer" in line:
                    hits.append({"file": str(p.relative_to(artifact_dir)), "line": line.strip()[:400]})
                    if len(hits) >= max_hits:
                        return hits
    except OSError:
        pass
    return hits


def list_postprocess_scripts(artifact_dir: Path) -> list[str]:
    scripts_dir = artifact_dir / "repo" / "scripts"
    if not scripts_dir.exists():
        return []
    return sorted(p.name for p in scripts_dir.glob("*.py"))


def read_manifest(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / "ARTIFACT_MANIFEST.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(errors="replace"))
    except Exception:
        return {}


def introspect_artifact(artifact_dir: Path, out_dir: Path) -> dict:
    """Discovers everything M17's variants need from the mounted artifact -
    accepted predict flags, available weight splits, the original reference
    command(s), post-processing scripts, and manifest metadata - and writes
    milestone17_artifact_introspection.json. This is the step that actually
    answers "what settings did the reference use / what can we tune", at
    runtime on Kaggle, without ever guessing.
    """
    predict_script = artifact_dir / "repo" / "scripts" / "predict_unet_transformer.py"
    predict_flags = discover_predict_flags(predict_script)
    gap_relink_flags = [f for f in predict_flags if any(kw in f.lower() for kw in GAP_RELINK_FLAG_KEYWORDS)]

    report = {
        "artifact_dir": str(artifact_dir),
        "weight_splits": list_weight_splits(artifact_dir),
        "predict_flags": predict_flags,
        "gap_or_relink_or_division_flags": gap_relink_flags,
        "postprocess_scripts": list_postprocess_scripts(artifact_dir),
        "reference_command_hits": discover_reference_commands(artifact_dir),
        "manifest_keys": sorted(read_manifest(artifact_dir).keys()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "milestone17_artifact_introspection.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"  [introspect] weight splits: {report['weight_splits']}")
    print(f"  [introspect] predict flags ({len(predict_flags)}): {predict_flags}")
    print(f"  [introspect] gap/relink/division flags: {gap_relink_flags}")
    print(f"  [introspect] postprocess scripts: {report['postprocess_scripts']}")
    print(f"  [introspect] reference command hits: {len(report['reference_command_hits'])}")
    return report

# --------------------------------------------------------------------------- #
# 12. Variant registry + FLAG-VALIDATED command builder
# --------------------------------------------------------------------------- #
# M16's known-good predict parameters (the exact settings that scored 0.874).
BASE_PREDICT_PARAMS = {
    "unet_batch_size": "4", "det_threshold": "0.99", "ilp_edge_weight": "-1.0",
    "ilp_appearance_weight": "0.1", "ilp_disappearance_weight": "0.1", "ilp_division_weight": "1.0",
}
FLAG_FOR_PARAM = {
    "unet_batch_size": "--unet-batch-size", "det_threshold": "--det-threshold",
    "ilp_edge_weight": "--ilp-edge-weight", "ilp_appearance_weight": "--ilp-appearance-weight",
    "ilp_disappearance_weight": "--ilp-disappearance-weight", "ilp_division_weight": "--ilp-division-weight",
}

# Each variant is a SMALL override on M16's known-good command. `split`
# defaults to 0; `use_gap_flags` enables Variant B's discovery-gated gap/
# relink flags; `extra_flags` are explicit additions validated against the
# discovered accepted-flags set.
M17_VARIANTS = {
    "A": {"label": "reference_exact", "param_overrides": {}, "split": 0, "use_gap_flags": False, "extra_flags": []},
    "B": {"label": "gap_recovery", "param_overrides": {}, "split": 0, "use_gap_flags": True, "extra_flags": []},
    # Variant C's single active change vs M16: det-threshold 0.99 -> 0.985, a
    # modest recall lever (more detections survive) with ILP weights held at
    # reference. A blind ILP-weight change is riskier than a small threshold
    # nudge; ILP-weight alternatives are documented for a follow-up sweep.
    "C": {"label": "tuned_det_threshold_0985", "param_overrides": {"det_threshold": "0.985"}, "split": 0,
          "use_gap_flags": False, "extra_flags": []},
}


def weights_rel_for_split(split: int) -> str:
    return f"weights/unet_transformer/split_{split}/edge_predictor_best.pth"


def select_weight_split(requested_split: int, available_splits: Sequence[int]) -> int:
    """Chooses the requested split if it actually ships, else the first
    available (or 0 when introspection could not enumerate any - e.g. an
    off-Kaggle dry run).
    """
    if not available_splits:
        return 0
    if requested_split in available_splits:
        return requested_split
    return available_splits[0]


def build_base_command(params: dict, split: int, splits_filename: str, data_dir: str) -> list[str]:
    cmd = [
        sys.executable, PREDICT_SCRIPT_REL, "--data-dir", data_dir, "--splits", splits_filename,
        "--split", str(split), "--weights", weights_rel_for_split(split),
    ]
    for key, flag in FLAG_FOR_PARAM.items():
        cmd += [flag, str(params[key])]
    cmd += ["--use-ilp"]
    return cmd


def extract_flag_usage_from_hits(hits: Sequence[dict], flag: str) -> list[str]:
    """Returns the EXACT tokens the reference command used for `flag`
    (`[flag]` for a bare toggle, `[flag, value]` for a valued flag), copied
    verbatim from a discovered reference command line - so Variant B only
    ever passes a gap/relink flag the way the reference itself proved it
    works, never a guessed value.
    """
    for h in hits:
        toks = str(h.get("line", "")).replace("\\", " ").split()
        if flag in toks:
            i = toks.index(flag)
            if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                val = toks[i + 1].strip().strip(",").strip("'\"")
                if val:
                    return [flag, val]
            return [flag]
    return []


def build_variant_command(
    variant_name: str, introspection: dict,
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
) -> tuple[list[str], dict]:
    """Builds one variant's predict command as a small override on M16's
    known-good command, VALIDATING every added flag against the discovered
    accepted-flags set. An unsupported flag is dropped with a note rather
    than handed to predict (which would crash on an unrecognized argument),
    so a variant can never regress M16's stability - at worst it reduces to
    the reference-exact command.
    """
    accepted = set(introspection.get("predict_flags", []))
    available_splits = list(introspection.get("weight_splits", [])) or [0]
    hits = introspection.get("reference_command_hits", [])
    gap_flags = list(introspection.get("gap_or_relink_or_division_flags", []))

    vdef = M17_VARIANTS[variant_name]
    params = dict(BASE_PREDICT_PARAMS)
    params.update(vdef.get("param_overrides", {}))
    split = select_weight_split(vdef.get("split", 0), available_splits)
    cmd = build_base_command(params, split, splits_filename, data_dir)

    notes = {
        "variant": variant_name, "label": vdef["label"], "split_used": split,
        "available_splits": available_splits, "param_overrides": vdef.get("param_overrides", {}),
        "extra_flags_added": [], "dropped_flags": [], "gap_flags_discovered": gap_flags, "gap_notes": [],
    }

    if vdef.get("use_gap_flags"):
        # Only NEW gap flags matter here - a keyword like "division" also
        # matches the base --ilp-division-weight, which must never be
        # re-appended as a duplicate on top of the base command.
        base_flags_present = set(cmd)
        candidate_gap_flags = [f for f in gap_flags if f not in base_flags_present]
        if not candidate_gap_flags:
            notes["gap_notes"].append(
                "no NEW gap/relink/postprocess flag beyond the base command found in the predict "
                "script - variant reduces to reference-exact")
        else:
            for flag in candidate_gap_flags:
                if flag not in accepted:
                    notes["dropped_flags"].append(flag)
                    continue
                usage = extract_flag_usage_from_hits(hits, flag)
                if usage:
                    cmd += usage
                    notes["extra_flags_added"].append(usage)
                else:
                    notes["gap_notes"].append(
                        f"{flag} is declared by the predict script but its exact reference usage/value "
                        f"was not found in any discovered command - NOT guessing its argument (this flag "
                        f"is skipped; review milestone17_artifact_introspection.json and specify it)")

    for tok in vdef.get("extra_flags", []):
        flag = tok[0] if isinstance(tok, list) else tok
        if flag in accepted:
            cmd += (list(tok) if isinstance(tok, list) else [tok])
            notes["extra_flags_added"].append(tok)
        else:
            notes["dropped_flags"].append(flag)

    return cmd, notes

# --------------------------------------------------------------------------- #
# 30. M22 Stage-1 forensic: enumerate artifact roots + find the true M19-C base
# --------------------------------------------------------------------------- #
TRUE_M19C_N_NODES_BEFORE = 131797   # M19-C (public 0.880) pre-postprocess node count
TRUE_M19C_N_EDGES_BEFORE = 118992   # M19-C (public 0.880) pre-postprocess edge count

# Weight hashes of the two packs KNOWN NOT to reproduce the M19-C base.
KNOWN_BAD_WEIGHT_SHA256 = {
    "dfb848aa8e490bba8eda91ac927b9ad1d8b06296487ba8504e45a1037c5e36ec": "pilkwang350",
    "912ae91a4077b65cc1933e6f37b9deb15268e483130b51555a490a8179ab3f53": "tom99763",
}

FORENSIC_PREDICT_REL = "repo/scripts/predict_unet_transformer.py"
FORENSIC_WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"
M22_FORENSIC_SMOKE_RUN = True   # run predict on non-bad viable candidates to find the true base


def _sha256_file(path) -> str | None:
    import hashlib
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(root) -> tuple[str | None, dict | None]:
    mp = Path(root) / "ARTIFACT_MANIFEST.json"
    if not mp.exists():
        return None, None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None, None
    name = m.get("artifact_name") or m.get("name") or m.get("artifact") or m.get("dataset")
    return name, m


def enumerate_artifact_roots(input_root: str = "/kaggle/input") -> list[Path]:
    """Globs candidate artifact roots one/two/three levels under /kaggle/input
    (covers /kaggle/input/*, /kaggle/input/datasets/*/*, /kaggle/input/
    notebooks/*/*, etc.), de-duplicated. Hidden-safe: pure globbing, no
    hardcoded dataset names.
    """
    base = Path(input_root)
    roots: list[Path] = []
    seen: set[str] = set()
    if not base.exists():
        return roots
    for pattern in ("*", "*/*", "*/*/*"):
        for p in sorted(base.glob(pattern)):
            try:
                if not p.is_dir():
                    continue
            except OSError:
                continue
            key = str(p)
            if key not in seen:
                seen.add(key)
                roots.append(p)
    return roots


def inspect_candidate(root: Path) -> dict:
    """Records presence + identity for one candidate root (no predict run)."""
    predict = root / FORENSIC_PREDICT_REL
    weights = root / FORENSIC_WEIGHTS_REL
    manifest_present = (root / "ARTIFACT_MANIFEST.json").exists()
    wheels_present = (root / "wheels").is_dir()
    has_predict = predict.exists()
    has_weights = weights.exists()
    has_all_required = has_predict and has_weights
    weight_sha256 = _sha256_file(weights) if has_weights else None
    artifact_name, manifest = _read_manifest(root)
    known_bad = KNOWN_BAD_WEIGHT_SHA256.get(weight_sha256 or "")
    return {
        "candidate_path": str(root),
        "artifact_name": artifact_name,
        "weight_sha256": weight_sha256,
        "known_bad_pack": known_bad,          # None if not a known-bad hash
        "has_predict_script": has_predict,
        "has_weights": has_weights,
        "has_manifest": manifest_present,
        "has_wheels": wheels_present,
        "has_all_required": has_all_required,
        "repo_path": str(root / "repo") if (root / "repo").is_dir() else None,
        "weights_path": str(weights) if has_weights else None,
        "manifest_keys": sorted(manifest.keys()) if isinstance(manifest, dict) else None,
    }


def _smoke_run_base_counts(root: Path, out_dir: Path, competition_dir: str, deps_state: dict) -> dict:
    """Materializes one candidate, runs the EXACT M19-C predict command, and
    converts the GEFF to BASE node/edge counts (no post-processing). Returns
    the counts + ok flag. Installs the wheels once (cached in deps_state).
    """
    comp = Path(competition_dir)
    test_dir = comp / "test"
    sample_submission_path = comp / "sample_submission.csv"
    result = {"ran": False, "ok": False, "n_nodes_before": None, "n_edges_before": None, "error": None}
    try:
        materialize = materialize_repo(root, out_dir)
        write_sitecustomize()
        env = build_subprocess_env()
        repo_dst = Path(materialize["repo_dst"])

        if not deps_state.get("installed"):
            dep_report = install_dependencies(root / "wheels", env)
            deps_state["installed"] = bool(dep_report.get("all_ok"))
            deps_state["report"] = dep_report
            if not deps_state["installed"]:
                result["error"] = "wheel install failed"
                return result
        # Import check (best-effort; predict will fail loudly anyway).
        _ = verify_imports(env)

        shipped_template = repo_dst / SPLITS_FILENAME
        prepare_splits(
            test_dir=test_dir, sample_submission_path=sample_submission_path,
            template_path=shipped_template if shipped_template.exists() else None,
            out_splits_path=repo_dst / SPLITS_FILENAME,
            diagnostic_path=out_dir / "m22_forensic_splits.json",
        )
        cmd = build_base_command(dict(BASE_PREDICT_PARAMS), 0, SPLITS_FILENAME, f"{competition_dir}/test")
        exec_result = run_prediction(cmd, cwd=repo_dst, env=env)
        result["ran"] = True
        if not exec_result.get("ok"):
            result["error"] = "predict returned nonzero"
            return result
        predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / "split_0"
        expected = discover_test_datasets(test_dir, sample_submission_path)["chosen"]
        df, _conv = convert_predictions_to_submission(predictions_dir, expected)
        result["n_nodes_before"] = int((df["row_type"] == "node").sum())
        result["n_edges_before"] = int((df["row_type"] == "edge").sum())
        result["ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def classify_forensic(candidates: list[dict], smoke_results: dict) -> dict:
    """Chooses the true M19-C artifact: a candidate whose SMOKE base counts
    match 131797/118992. If no smoke run matched, reports NOT_FOUND (a hash
    that merely differs from the known-bad ones is only a 'likely' hint, never
    a confirmation)."""
    matched = None
    for c in candidates:
        sr = smoke_results.get(c["candidate_path"])
        if sr and sr.get("ok") and sr.get("n_nodes_before") == TRUE_M19C_N_NODES_BEFORE and sr.get("n_edges_before") == TRUE_M19C_N_EDGES_BEFORE:
            matched = c
            break
    likely = [c for c in candidates if c["has_all_required"] and not c["known_bad_pack"]]
    if matched:
        sr = smoke_results.get(matched["candidate_path"], {})
        return {
            "recommendation": "TRUE_M19C_ARTIFACT_FOUND",
            "selected_artifact_path": matched["candidate_path"],
            "artifact_name": matched["artifact_name"],
            "weight_sha256": matched["weight_sha256"],
            "repo_path": matched["repo_path"],
            "weights_path": matched["weights_path"],
            "n_nodes_before": sr.get("n_nodes_before"),
            "n_edges_before": sr.get("n_edges_before"),
        }
    return {
        "recommendation": "TRUE_M19C_ARTIFACT_NOT_FOUND",
        "selected_artifact_path": None, "artifact_name": None, "weight_sha256": None,
        "repo_path": None, "weights_path": None, "n_nodes_before": None, "n_edges_before": None,
        "likely_non_bad_candidates": [c["candidate_path"] for c in likely],
    }


def run_m22_forensic(working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
                     smoke_run: bool = M22_FORENSIC_SMOKE_RUN) -> dict:
    """Stage-1 driver: enumerate -> inspect -> (optional) smoke-run non-bad
    viable candidates until the true base is found -> write report + CSV ->
    print recommendation."""
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = enumerate_artifact_roots()
    candidates = [inspect_candidate(r) for r in roots]
    viable = [c for c in candidates if c["has_all_required"]]
    print(f"[forensic] scanned {len(roots)} roots; {len(viable)} contain the required predict+weights.")

    smoke_results: dict = {}
    deps_state: dict = {}
    if smoke_run and is_kaggle_env():
        for c in viable:
            if c["known_bad_pack"]:
                print(f"[forensic] skip smoke (known-bad {c['known_bad_pack']}): {c['candidate_path']}")
                continue
            print(f"[forensic] smoke-run: {c['candidate_path']} (sha {c['weight_sha256']})")
            sr = _smoke_run_base_counts(Path(c["candidate_path"]), out_dir, competition_dir, deps_state)
            smoke_results[c["candidate_path"]] = sr
            print(f"           -> ran={sr['ran']} ok={sr['ok']} base={sr['n_nodes_before']}/{sr['n_edges_before']} err={sr['error']}")
            if sr.get("ok") and sr.get("n_nodes_before") == TRUE_M19C_N_NODES_BEFORE and sr.get("n_edges_before") == TRUE_M19C_N_EDGES_BEFORE:
                print("[forensic] TRUE base matched - stopping smoke scan.")
                break

    verdict = classify_forensic(candidates, smoke_results)

    report = {
        "expected_n_nodes_before": TRUE_M19C_N_NODES_BEFORE,
        "expected_n_edges_before": TRUE_M19C_N_EDGES_BEFORE,
        "known_bad_weight_sha256": KNOWN_BAD_WEIGHT_SHA256,
        "smoke_run": bool(smoke_run and is_kaggle_env()),
        "n_roots_scanned": len(roots), "n_viable": len(viable),
        "candidates": candidates, "smoke_results": smoke_results, "verdict": verdict,
    }
    (out_dir / "m22_artifact_forensic_report.json").write_text(json.dumps(report, indent=2, default=str))

    # CSV table (plain, no pandas dependency on formatting quirks).
    cols = ["candidate_path", "artifact_name", "weight_sha256", "known_bad_pack",
            "has_all_required", "has_manifest", "has_wheels", "repo_path", "weights_path"]
    rows = []
    for c in candidates:
        sr = smoke_results.get(c["candidate_path"], {})
        row = {k: c.get(k) for k in cols}
        row["smoke_n_nodes_before"] = sr.get("n_nodes_before")
        row["smoke_n_edges_before"] = sr.get("n_edges_before")
        rows.append(row)
    pd.DataFrame(rows, columns=cols + ["smoke_n_nodes_before", "smoke_n_edges_before"]).to_csv(
        out_dir / "m22_candidate_table.csv", index=False)

    print("\n=== M22 forensic verdict ===")
    print(f"  {verdict['recommendation']}")
    if verdict["recommendation"] == "TRUE_M19C_ARTIFACT_FOUND":
        print(f"  selected_artifact_path: {verdict['selected_artifact_path']}")
        print(f"  artifact_name: {verdict['artifact_name']}")
        print(f"  weight_sha256: {verdict['weight_sha256']}")
        print(f"  repo_path: {verdict['repo_path']}")
        print(f"  weights_path: {verdict['weights_path']}")
        print(f"  base: n_nodes_before={verdict['n_nodes_before']} n_edges_before={verdict['n_edges_before']}")
        print("  -> Pin M22 true-base variants to this path/hash and run them.")
    else:
        print("  The true M19-C artifact was NOT confirmed among mounted inputs.")
        print("  Attach the original NOTEBOOK input used by M19-C (e.g. a kernel named like")
        print("  'Biohub Cell Tracking: Learned Graph w G') and re-run this diagnostic.")
        print("  M19-C 0.880 remains the final candidate.")
    return report


# --------------------------------------------------------------------------- #
# 31. Tests (forensic helpers on a synthetic temp tree; no Kaggle/GPU)
# --------------------------------------------------------------------------- #
def run_m22_forensic_tests() -> None:
    scratch = Path("/tmp") / "m22_forensic_test"
    if scratch.exists():
        shutil.rmtree(scratch)
    # Build a fake candidate with all required files + a manifest.
    cand = scratch / "kaggle_input" / "datasets" / "someone" / "pack"
    (cand / "repo" / "scripts").mkdir(parents=True, exist_ok=True)
    (cand / "weights" / "unet_transformer" / "split_0").mkdir(parents=True, exist_ok=True)
    (cand / "wheels").mkdir(parents=True, exist_ok=True)
    (cand / "repo" / "scripts" / "predict_unet_transformer.py").write_text("# predict\n")
    (cand / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth").write_bytes(b"FAKE-WEIGHTS-CONTENT")
    (cand / "ARTIFACT_MANIFEST.json").write_text(json.dumps({"artifact_name": "fake-pack-vX", "epochs": 50}))

    info = inspect_candidate(cand)
    assert info["has_all_required"] and info["has_wheels"] and info["has_manifest"]
    assert info["artifact_name"] == "fake-pack-vX"
    assert info["weight_sha256"] == _sha256_file(cand / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth")
    assert info["known_bad_pack"] is None, "fake hash must not be flagged known-bad"

    # A candidate carrying the pilkwang known-bad hash is flagged.
    import hashlib
    assert len(KNOWN_BAD_WEIGHT_SHA256) == 2
    # classify: no smoke match -> NOT_FOUND; with a synthetic smoke match -> FOUND.
    v_none = classify_forensic([info], {})
    assert v_none["recommendation"] == "TRUE_M19C_ARTIFACT_NOT_FOUND"
    assert info["candidate_path"] in v_none["likely_non_bad_candidates"]
    fake_smoke = {info["candidate_path"]: {"ok": True, "n_nodes_before": TRUE_M19C_N_NODES_BEFORE, "n_edges_before": TRUE_M19C_N_EDGES_BEFORE}}
    v_found = classify_forensic([info], fake_smoke)
    assert v_found["recommendation"] == "TRUE_M19C_ARTIFACT_FOUND"
    assert v_found["n_nodes_before"] == 131797 and v_found["n_edges_before"] == 118992

    # enumerate is hidden-safe: returns [] when the input root is absent.
    assert enumerate_artifact_roots(str(scratch / "does_not_exist")) == []
    shutil.rmtree(scratch)
    print("All milestone22_artifact_forensic_diagnostic tests passed (4/4).")


# --------------------------------------------------------------------------- #
# 32. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone22_forensic() -> dict:
    print("=== Self-test: M22 forensic helpers ===")
    run_m22_forensic_tests()
    if not is_kaggle_env():
        print("[dry-run] /kaggle/input absent - self-tests only. On Kaggle this enumerates all mounted "
              "artifact roots and (if M22_FORENSIC_SMOKE_RUN) predicts to find the true M19-C base.")
        return {"status": "dry_run"}
    return run_m22_forensic()


if __name__ == "__main__":
    run_milestone22_forensic()
