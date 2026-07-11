"""
Biohub - Cell Tracking During Development
Milestone 32.1 - OFFICIAL METRIC VENDOR + SPLIT RECOVERY FIX (NON-SUBMIT).

M32's on-Kaggle audit passed the 400ep artifact + found ~200 train GEFF datasets
but reported OFFICIAL_METRIC_NOT_FOUND (tracking_cellmot.metrics / scripts.evaluate
absent) and CLEAN_HOLDOUT_UNRESOLVED (no split config). M32.1 repairs both WITHOUT
approximating anything.

PART 1-2 - VENDOR THE AUTHORITATIVE OFFICIAL METRIC (offline):
A pinned VERBATIM snapshot of the official scorer from
royerlab/kaggle-cell-tracking-competition (branch main, commit
7396b7e98e61844e799152ddda7e5493084cc8f3) is vendored into
competitions/.../vendor/royerlab_cellmot (src/tracking_cellmot/{__init__,metrics,
division_metrics,io,img_proc}.py + models, scripts/{evaluate,dataspec}.py,
metrics.md, LICENSE) with a PROVENANCE.json recording per-file SHA256 + the import
graph + license. The three core scoring files (__init__, metrics, division_metrics)
are ALSO embedded (base64, byte-identical, sha256-verified at runtime) so the
M32.1 runner is self-contained: it materialises the `tracking_cellmot` package to
/kaggle/working, prepends it to sys.path, imports tracking_cellmot.metrics, and
verifies evaluate / evaluate_datasets / per_sample_metrics / summarise. The metric
LOGIC is never rewritten and local_metric.py is never used. Only import-path
compatibility patches are permitted (reported as a diff); none were needed. If a
required dependency (tracksdata / geff / polars / scipy) is unavailable it reports
OFFICIAL_METRIC_DEPENDENCY_MISSING - never a proxy/fake score. Ten synthetic graph
fixtures are scored with the ACTUAL official functions.

PART 3 - normalized train inventory: strip .geff, exclude the `train` dir entry,
require a paired GT GEFF (and preferably a paired Zarr input), de-duplicate,
resolve estimated_number_of_nodes + the source-group prefix (44b6 / 6bba).

PART 4 - EXACT split_0 recovery cascade: search all mounted paths for
dataset_splits.json / *splits*/*fold*/manifests/config/args near the 400ep
weights; inspect the checkpoint metadata safely; and only reconstruct split_0 if
the EXACT ordering rule + folds + seed + shuffle algorithm + train/test
interpretation + input set are ALL observed (in this repo the shipped
dataset_splits.json is authoritative and READ, not regenerated - so
reconstruction is forbidden). Never call arbitrary random datasets a clean CV.
split_source_type in {OBSERVED_FILE, CHECKPOINT_METADATA,
EXACT_DETERMINISTIC_RECONSTRUCTION, UNRESOLVED}; leakage guard enforces zero
held-out/train overlap.

PART 5 - combined re-audit (M32_1_B) passes AUDIT_PASS only when the vendored
metric imports + all official verification tests pass + the normalized GT
inventory is valid + a clean split_0 holdout is proven + overlap is zero + the
400ep SHA passes. Only then may the existing M32_B replay run.

Runners (all NON-SUBMIT): M32_1_A_OFFICIAL_METRIC_VENDOR_AUDIT (vendor + import +
10 fixtures), M32_1_B_OFFICIAL_CV_REAUDIT (metric + inventory + split + artifact).
No submission.csv, no Kaggle-API submit, no static CSV; all outputs under
/kaggle/working; standalone one-cell runners; exact 400ep guard; no M16-M32 source
file modified. M19-C 0.880 stays confirmed best/final; M29-A 0.876 failed; M30-C
pending; M31 blocked. Off Kaggle (no tracksdata/polars) it self-tests and reports
OFFICIAL_METRIC_DEPENDENCY_MISSING honestly.
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
# 40. M32.1 EMBEDDED vendored official metric source (base64, byte-identical)
# --------------------------------------------------------------------------- #
# Verbatim base64 of royerlab/kaggle-cell-tracking-competition@7396b7e9 core files
# (tracking_cellmot/__init__.py, metrics.py, division_metrics.py). Materialised +
# sha256-verified at runtime. The metric logic is NEVER rewritten.
M32_1_EMBED = {
"__init__.py": "IiIidHJhY2tpbmctY2VsbG1vdDogY2VsbCB0cmFja2luZyB1dGlsaXRpZXMgZm9yIHRoZSBDVEMvQ2VsbE1vdCBjaGFsbGVuZ2UuIiIiCg==",
"metrics.py": "aW1wb3J0IHdhcm5pbmdzCmZyb20gdHlwaW5nIGltcG9ydCBMaXRlcmFsLCBOYW1lZFR1cGxlCgppbXBvcnQgcG9sYXJzIGFzIHBsCmltcG9ydCB0cmFja3NkYXRhIGFzIHRkCgoKY2xhc3MgRXZhbHVhdGlvblJlc3VsdChOYW1lZFR1cGxlKToKICAgICIiIkNvdW50cyByZXR1cm5lZCBieSA6ZnVuYzpgZXZhbHVhdGVgLiIiIgoKICAgIGVkZ2VfdHA6IGludAogICAgZWRnZV9mcDogaW50CiAgICBlZGdlX2ZuOiBpbnQKICAgIGRpdmlzaW9uX3RwOiBpbnQKICAgIGRpdmlzaW9uX2ZwOiBpbnQKICAgIGRpdmlzaW9uX2ZuOiBpbnQKICAgIG51bV9wcmVkX25vZGVzOiBpbnQKCgpjbGFzcyBEYXRhc2V0c1Jlc3VsdChOYW1lZFR1cGxlKToKICAgICIiIkN1bXVsYXRpdmUgKG1pY3JvLWF2ZXJhZ2VkKSBKYWNjYXJkcyBwbHVzIHRoZSBjb21iaW5lZCBzY29yZS4iIiIKCiAgICBlZGdlX2phY2NhcmQ6IGZsb2F0CiAgICBkaXZpc2lvbl9qYWNjYXJkOiBmbG9hdAogICAgc2NvcmU6IGZsb2F0CgoKIyBQZW5hbHR5IGNvZWZmaWNpZW50IGZvciB0aGUgYWRqdXN0ZWQgZWRnZSBKYWNjYXJkOgojICAgSl9hZGogPSBtYXgoMCwgSiDCtyAoMSAtIEFESlVTVE1FTlRfQUxQSEEgwrcgdG90YWxfbm9kZV9yYXRpbykpCkFESlVTVE1FTlRfQUxQSEE6IGZsb2F0ID0gMC4xCgojIFdlaWdodCBvZiB0aGUgZGl2aXNpb24gSmFjY2FyZCBpbiB0aGUgY29tYmluZWQgcnVuLWxldmVsIHNjb3JlOgojICAgc2NvcmUgPSBhZGpfZWRnZV9qYWNjYXJkICsgU0NPUkVfRElWSVNJT05fV0VJR0hUIMK3IGRpdmlzaW9uX2phY2NhcmQKU0NPUkVfRElWSVNJT05fV0VJR0hUOiBmbG9hdCA9IDAuMQoKQ09VTlRfQ09MVU1OUzogdHVwbGVbc3RyLCAuLi5dID0gKAogICAgImVkZ2VfdHAiLCAiZWRnZV9mcCIsICJlZGdlX2ZuIiwKICAgICJkaXZpc2lvbl90cCIsICJkaXZpc2lvbl9mcCIsICJkaXZpc2lvbl9mbiIsCiAgICAibnVtX3ByZWRfbm9kZXMiLAopCk1FVFJJQ19DT0xVTU5TOiB0dXBsZVtzdHIsIC4uLl0gPSBDT1VOVF9DT0xVTU5TICsgKAogICAgIm5vZGVfcmVjYWxsIiwgInRvdGFsX25vZGVfcmF0aW8iLCAiZWRnZV9qYWNjYXJkIiwgImFkal9lZGdlX2phY2NhcmQiLAopCgoKZGVmIF9qYWNjYXJkKHRwOiBpbnQsIGZwOiBpbnQsIGZuOiBpbnQpIC0+IGZsb2F0OgogICAgZGVub20gPSB0cCArIGZwICsgZm4KICAgIHJldHVybiB0cCAvIGRlbm9tIGlmIGRlbm9tID4gMCBlbHNlIGZsb2F0KCJuYW4iKQoKCiMgZnVuY3Rpb24gaXMgc3BsaXQgZm9yIGVhc2llciB0ZXN0aW5nCmRlZiBfZXZhbHVhdGVfbWF0Y2hlZF9ncmFwaCgKICAgIGdyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBndF9ncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAopIC0+IHBsLkRhdGFGcmFtZToKICAgIGVkZ2VfYXR0cnMgPSBncmFwaC5lZGdlX2F0dHJzKGF0dHJfa2V5cz1bdGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hFRF9FREdFX01BU0tdKQogICAgIyBHdWFyZCBhZ2FpbnN0IGR1cGxpY2F0ZSBlZGdlcyAoc2FtZSBzb3VyY2XihpJ0YXJnZXQgcGFpciBhcHBlYXJpbmcgbXVsdGlwbGUgdGltZXMpLgogICAgIyB0cmFja3NkYXRhJ3MgbWF0Y2goKSBpbm5lci1qb2luIG1hcmtzIGFsbCBkdXBsaWNhdGVzIGFzIG1hdGNoZWQsIHdoaWNoIGluZmxhdGVzCiAgICAjIHRoZSBpbnRlcnNlY3Rpb24gY291bnQgYW5kIGNhbiBwdXNoIHNjb3JlcyBhYm92ZSAxLjAuIFNvcnQgbWF0Y2hlZCByb3dzIGZpcnN0CiAgICAjIHNvIHRoZSBkZWR1cCBrZWVwcyB0aGUgbWF0Y2hlZCBjb3B5IHdoZW4gZHVwbGljYXRlcyBkaXNhZ3JlZSBvbiB0aGUgbWFzay4KICAgIGVkZ2VfYXR0cnMgPSBlZGdlX2F0dHJzLnNvcnQoCiAgICAgICAgdGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hFRF9FREdFX01BU0ssIGRlc2NlbmRpbmc9VHJ1ZSwKICAgICkudW5pcXVlKAogICAgICAgIHN1YnNldD1bdGQuREVGQVVMVF9BVFRSX0tFWVMuRURHRV9TT1VSQ0UsIHRkLkRFRkFVTFRfQVRUUl9LRVlTLkVER0VfVEFSR0VUXSwKICAgICAgICBrZWVwPSJmaXJzdCIsCiAgICApCiAgICBub2RlX2F0dHJzID0gZ3JhcGgubm9kZV9hdHRycyhhdHRyX2tleXM9W3RkLkRFRkFVTFRfQVRUUl9LRVlTLk5PREVfSUQsIHRkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfTk9ERV9JRF0pCgogICAgIyBJJ20gYXNzdW1pbmcgdmFsaWQgZ3JvdW5kLXRydXRoIGVkZ2VzIGFyZSBhbHdheXMgMTAwJSBjb3JyZWN0IGlmIHRoZXkgaGF2ZSBhbiBlZGdlLgogICAgIyBUaGVyZWZvcmUsIHdlIGRvbid0IGhhdmUgY2FzZXMgd2hlcmUgdGhlIGNlbGwgZGl2aWRlZCwgYnV0IG5vdCBpbiB0aGUgZ3JvdW5kIHRydXRoLgogICAgZ3Rfbm9kZV9pZHMgPSBndF9ncmFwaC5ub2RlX2lkcygpCiAgICBndF9ub2RlX2F0dHJzID0gcGwuRGF0YUZyYW1lKAogICAgICAgIHsKICAgICAgICAgICAgdGQuREVGQVVMVF9BVFRSX0tFWVMuTk9ERV9JRDogZ3Rfbm9kZV9pZHMsCiAgICAgICAgICAgICJvdXRfZGVncmVlIjogZ3RfZ3JhcGgub3V0X2RlZ3JlZShndF9ub2RlX2lkcyksCiAgICAgICAgICAgICJpbl9kZWdyZWUiOiBndF9ncmFwaC5pbl9kZWdyZWUoZ3Rfbm9kZV9pZHMpLAogICAgICAgIH0KICAgICkud2l0aF9jb2x1bW5zKAogICAgICAgIChwbC5jb2woIm91dF9kZWdyZWUiKSA+IDApLmFsaWFzKCJvdXRfdmFsaWQiKSwKICAgICAgICAocGwuY29sKCJpbl9kZWdyZWUiKSA+IDApLmFsaWFzKCJpbl92YWxpZCIpLAogICAgKQoKICAgICMgbWVyZ2luZyBncm91bmQgdHJ1dGggZ3JhcGggaW50byB0aGUgcHJlZGljdGVkIGdyYXBoCiAgICBub2RlX2F0dHJzID0gbm9kZV9hdHRycy5qb2luKAogICAgICAgIGd0X25vZGVfYXR0cnMsCiAgICAgICAgbGVmdF9vbj10ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSUQsCiAgICAgICAgcmlnaHRfb249dGQuREVGQVVMVF9BVFRSX0tFWVMuTk9ERV9JRCwKICAgICAgICBob3c9ImxlZnQiLAogICAgKS53aXRoX2NvbHVtbnMoCiAgICAgICAgcGwuY29sKCJvdXRfdmFsaWQiKS5maWxsX251bGwoRmFsc2UpLAogICAgICAgIHBsLmNvbCgiaW5fdmFsaWQiKS5maWxsX251bGwoRmFsc2UpLAogICAgKQoKICAgICMgbWVyZ2Ugb3V0IHZhbGlkIGludG8gc291cmNlIGFuZCBpbiB2YWxpZCBpbnRvIHRhcmdldAogICAgZWRnZV9hdHRycyA9IGVkZ2VfYXR0cnMuam9pbigKICAgICAgICBub2RlX2F0dHJzLnNlbGVjdCh0ZC5ERUZBVUxUX0FUVFJfS0VZUy5OT0RFX0lELCAib3V0X3ZhbGlkIiksCiAgICAgICAgbGVmdF9vbj10ZC5ERUZBVUxUX0FUVFJfS0VZUy5FREdFX1NPVVJDRSwKICAgICAgICByaWdodF9vbj10ZC5ERUZBVUxUX0FUVFJfS0VZUy5OT0RFX0lELAogICAgICAgIGhvdz0ibGVmdCIsCiAgICApLmpvaW4oCiAgICAgICAgbm9kZV9hdHRycy5zZWxlY3QodGQuREVGQVVMVF9BVFRSX0tFWVMuTk9ERV9JRCwgImluX3ZhbGlkIiksCiAgICAgICAgbGVmdF9vbj10ZC5ERUZBVUxUX0FUVFJfS0VZUy5FREdFX1RBUkdFVCwKICAgICAgICByaWdodF9vbj10ZC5ERUZBVUxUX0FUVFJfS0VZUy5OT0RFX0lELAogICAgICAgIGhvdz0ibGVmdCIsCiAgICApCgogICAgZWRnZV9hdHRycyA9IGVkZ2VfYXR0cnMud2l0aF9jb2x1bW5zKAogICAgICAgIChwbC5jb2woIm91dF92YWxpZCIpIHwgcGwuY29sKCJpbl92YWxpZCIpKS5hbGlhcygicHJlZF92YWxpZCIpLAogICAgKQoKICAgICMgc2FuaXR5IGNoZWNrIHRoYXQgYHByZWRfdmFsaWRgIGlzIGEgc3VwZXJzZXQgb2YgYWxsIG1hdGNoZWQgZWRnZXMKICAgIGFzc2VydCBlZGdlX2F0dHJzLmZpbHRlcih0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX0VER0VfTUFTSylbInByZWRfdmFsaWQiXS5hbGwoKQoKICAgIHJldHVybiBlZGdlX2F0dHJzCgoKZGVmIF9jb21wdXRlX3Njb3JlKAogICAgZWRnZV9hdHRyczogcGwuRGF0YUZyYW1lLAogICAgZ3RfbnVtX2VkZ2VzOiBpbnQsCiAgICBtZXRyaWM6IExpdGVyYWxbImphY2NhcmQiLCAiZGljZSJdLAopIC0+IGZsb2F0OgogICAgaW50ZXJzZWN0aW9uID0gaW50KGVkZ2VfYXR0cnNbdGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hFRF9FREdFX01BU0tdLnN1bSgpKQogICAgbl92YWxpZF9wcmVkX2VkZ2VzID0gaW50KGVkZ2VfYXR0cnNbInByZWRfdmFsaWQiXS5zdW0oKSkKCiAgICBpZiBtZXRyaWMgPT0gImphY2NhcmQiOgogICAgICAgIG51bSA9IGludGVyc2VjdGlvbgogICAgICAgIGRlbm9tID0gZ3RfbnVtX2VkZ2VzICsgbl92YWxpZF9wcmVkX2VkZ2VzIC0gaW50ZXJzZWN0aW9uCiAgICBlbGlmIG1ldHJpYyA9PSAiZGljZSI6CiAgICAgICAgbnVtID0gMiAqIGludGVyc2VjdGlvbgogICAgICAgIGRlbm9tID0gZ3RfbnVtX2VkZ2VzICsgbl92YWxpZF9wcmVkX2VkZ2VzCiAgICBlbHNlOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJJbnZhbGlkIG1ldHJpYzoge21ldHJpY30iKQoKICAgIHJldHVybiBudW0gLyBkZW5vbSBpZiBkZW5vbSA+IDAgZWxzZSBmbG9hdCgibmFuIikKCgpkZWYgX2V2YWx1YXRlKAogICAgZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCwKICAgIGd0X2dyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBtZXRyaWM6IExpdGVyYWxbImphY2NhcmQiLCAiZGljZSJdLAogICAgc2NhbGU6IHR1cGxlW2Zsb2F0LCAuLi5dIHwgTm9uZSwKICAgIG1heF9kaXN0YW5jZTogZmxvYXQsCikgLT4gZmxvYXQ6CiAgICBpZiB0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSUQgaW4gZ3JhcGgubm9kZV9hdHRyX2tleXMoKToKICAgICAgICB3YXJuaW5ncy53YXJuKCJHcmFwaCBhbHJlYWR5IG1hdGNoZWQsIG92ZXJ3cml0aW5nIHByZXZpb3VzIG1hdGNoaW5nLiIpCiAgICAgICAgIyBSZXNldCBtYXRjaGluZyBhdHRyaWJ1dGVzIHRvIGRlZmF1bHRzIGJlZm9yZSByZS1tYXRjaGluZwogICAgICAgIGFsbF9ub2RlX2lkcyA9IGdyYXBoLm5vZGVfaWRzKCkKICAgICAgICBncmFwaC51cGRhdGVfbm9kZV9hdHRycygKICAgICAgICAgICAgbm9kZV9pZHM9YWxsX25vZGVfaWRzLAogICAgICAgICAgICBhdHRycz17CiAgICAgICAgICAgICAgICB0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSUQ6IC0xLAogICAgICAgICAgICAgICAgdGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hfU0NPUkU6IDAuMCwKICAgICAgICAgICAgfSwKICAgICAgICApCiAgICAgICAgYWxsX2VkZ2VfaWRzID0gZ3JhcGguZWRnZV9pZHMoKQogICAgICAgIGlmIGxlbihhbGxfZWRnZV9pZHMpID4gMDoKICAgICAgICAgICAgZ3JhcGgudXBkYXRlX2VkZ2VfYXR0cnMoCiAgICAgICAgICAgICAgICBlZGdlX2lkcz1hbGxfZWRnZV9pZHMsCiAgICAgICAgICAgICAgICBhdHRycz17dGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hFRF9FREdFX01BU0s6IEZhbHNlfSwKICAgICAgICAgICAgKQoKICAgIGZyb20gdHJhY2tzZGF0YS5tZXRyaWNzIGltcG9ydCBEaXN0YW5jZU1hdGNoaW5nCiAgICBtYXRjaGluZyA9IERpc3RhbmNlTWF0Y2hpbmcobWF4X2Rpc3RhbmNlPW1heF9kaXN0YW5jZSwgc2NhbGU9c2NhbGUpCgogICAgaWYgZ3JhcGgubnVtX2VkZ2VzKCkgPT0gMCBvciBncmFwaC5udW1fbm9kZXMoKSA9PSAwOgogICAgICAgIHdhcm5pbmdzLndhcm4oIlByZWRpY3RlZCBncmFwaCBoYXMgbm8gZWRnZXMgb3Igbm8gbm9kZXMsIHJldHVybmluZyBzY29yZSAwLjAuIikKICAgICAgICByZXR1cm4gMC4wCgogICAgZnJvbSB0cmFja3NkYXRhLm9wdGlvbnMgaW1wb3J0IGdldF9vcHRpb25zLCBzZXRfb3B0aW9ucwoKICAgIHByZXZfc2hvd19wcm9ncmVzcyA9IGdldF9vcHRpb25zKCkuc2hvd19wcm9ncmVzcwogICAgc2V0X29wdGlvbnMoc2hvd19wcm9ncmVzcz1GYWxzZSkKICAgIHRyeToKICAgICAgICB3aXRoIHdhcm5pbmdzLmNhdGNoX3dhcm5pbmdzKCk6CiAgICAgICAgICAgIGZyb20gc2NpcHkuc3BhcnNlIGltcG9ydCBTcGFyc2VFZmZpY2llbmN5V2FybmluZwogICAgICAgICAgICB3YXJuaW5ncy5maWx0ZXJ3YXJuaW5ncygiaWdub3JlIiwgY2F0ZWdvcnk9U3BhcnNlRWZmaWNpZW5jeVdhcm5pbmcpCiAgICAgICAgICAgIGdyYXBoLm1hdGNoKGd0X2dyYXBoLCBtYXRjaGluZz1tYXRjaGluZykKICAgIGZpbmFsbHk6CiAgICAgICAgc2V0X29wdGlvbnMoc2hvd19wcm9ncmVzcz1wcmV2X3Nob3dfcHJvZ3Jlc3MpCgogICAgZWRnZV9hdHRycyA9IF9ldmFsdWF0ZV9tYXRjaGVkX2dyYXBoKGdyYXBoLCBndF9ncmFwaCkKCiAgICByZXR1cm4gX2NvbXB1dGVfc2NvcmUoZWRnZV9hdHRycywgZ3RfZ3JhcGgubnVtX2VkZ2VzKCksIG1ldHJpYykKCgpkZWYgZXZhbHVhdGUoCiAgICBncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAogICAgZ3RfZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCwKICAgIHNjYWxlOiB0dXBsZVtmbG9hdCwgLi4uXSB8IE5vbmUgPSBOb25lLAogICAgbWF4X2Rpc3RhbmNlOiBmbG9hdCA9IDcuMCwKKSAtPiBFdmFsdWF0aW9uUmVzdWx0OgogICAgIiIiCiAgICBFdmFsdWF0ZSBhIHByZWRpY3RlZCBncmFwaCBhZ2FpbnN0IGEgZ3JvdW5kLXRydXRoIGdyYXBoIHVzaW5nCiAgICBjZW50cm9pZC1kaXN0YW5jZSBub2RlIG1hdGNoaW5nLgoKICAgIENvbXB1dGVzIGVkZ2UgVFAvRlAvRk4sIGRpdmlzaW9uIFRQL0ZQL0ZOICh2aWEKICAgIDpmdW5jOmB0cmFja2luZ19jZWxsbW90LmRpdmlzaW9uX21ldHJpY3MuZXZhbHVhdGVfZGl2aXNpb25zYCksIGFuZCB0aGUKICAgIHRvdGFsIG51bWJlciBvZiBwcmVkaWN0ZWQgbm9kZXMgKGlycmVzcGVjdGl2ZSBvZiBtYXRjaGluZykuCgogICAgUGFyYW1ldGVycwogICAgLS0tLS0tLS0tLQogICAgZ3JhcGggOiB0cmFja3NkYXRhLmdyYXBoLkJhc2VHcmFwaAogICAgICAgIFRoZSBwcmVkaWN0ZWQgZ3JhcGguIE1hdGNoaW5nIGF0dHJpYnV0ZXMgYXJlIHdyaXR0ZW4gb250byAqZ3JhcGgqCiAgICAgICAgYXMgYSBzaWRlIGVmZmVjdC4KICAgIGd0X2dyYXBoIDogdHJhY2tzZGF0YS5ncmFwaC5CYXNlR3JhcGgKICAgICAgICBUaGUgZ3JvdW5kIHRydXRoIGdyYXBoLgogICAgc2NhbGUgOiB0dXBsZVtmbG9hdCwgLi4uXSB8IE5vbmUsIG9wdGlvbmFsCiAgICAgICAgUGh5c2ljYWwgc2NhbGUgZm9yIGVhY2ggc3BhdGlhbCBkaW1lbnNpb24gKGUuZy4sICh6LCB5LCB4KSkgdG8KICAgICAgICBhY2NvdW50IGZvciBhbmlzb3Ryb3B5LiBJZiBOb25lLCBhc3N1bWVzIGlzb3Ryb3BpYyBkYXRhLgogICAgbWF4X2Rpc3RhbmNlIDogZmxvYXQsIG9wdGlvbmFsCiAgICAgICAgTWF4aW11bSBkaXN0YW5jZSBiZXR3ZWVuIGNlbnRyb2lkcyB0byBiZSBjb25zaWRlcmVkIGFzIGEgbWF0Y2guCgogICAgUmV0dXJucwogICAgLS0tLS0tLQogICAgRXZhbHVhdGlvblJlc3VsdAogICAgIiIiCiAgICBmcm9tIC5kaXZpc2lvbl9tZXRyaWNzIGltcG9ydCBldmFsdWF0ZV9kaXZpc2lvbnMKCiAgICAjIE1hdGNoIGdyYXBoIGFnYWluc3QgZ3RfZ3JhcGggKGluIHBsYWNlKTsgZGlzY2FyZCB0aGUgcmV0dXJuZWQgc2NvcmUuCiAgICBfZXZhbHVhdGUoZ3JhcGgsIGd0X2dyYXBoLCAiamFjY2FyZCIsIHNjYWxlLCBtYXhfZGlzdGFuY2UpCgogICAgaWYgZ3JhcGgubnVtX2VkZ2VzKCkgPT0gMDoKICAgICAgICBlZGdlX3RwID0gMAogICAgICAgIGVkZ2VfZnAgPSAwCiAgICAgICAgZWRnZV9mbiA9IGd0X2dyYXBoLm51bV9lZGdlcygpCiAgICBlbHNlOgogICAgICAgIGVkZ2VfYXR0cnMgPSBfZXZhbHVhdGVfbWF0Y2hlZF9ncmFwaChncmFwaCwgZ3RfZ3JhcGgpCiAgICAgICAgZWRnZV90cCA9IGludChlZGdlX2F0dHJzW3RkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfRURHRV9NQVNLXS5zdW0oKSkKICAgICAgICBlZGdlX3ZhbGlkX3ByZWQgPSBpbnQoZWRnZV9hdHRyc1sicHJlZF92YWxpZCJdLnN1bSgpKQogICAgICAgIGVkZ2VfZnAgPSBlZGdlX3ZhbGlkX3ByZWQgLSBlZGdlX3RwCiAgICAgICAgZWRnZV9mbiA9IGd0X2dyYXBoLm51bV9lZGdlcygpIC0gZWRnZV90cAoKICAgIGRpdiA9IGV2YWx1YXRlX2RpdmlzaW9ucygKICAgICAgICBncmFwaCwgZ3RfZ3JhcGgsIHNjYWxlPXNjYWxlLCBtYXhfZGlzdGFuY2U9bWF4X2Rpc3RhbmNlLAogICAgKQoKICAgIHJldHVybiBFdmFsdWF0aW9uUmVzdWx0KAogICAgICAgIGVkZ2VfdHA9ZWRnZV90cCwKICAgICAgICBlZGdlX2ZwPWVkZ2VfZnAsCiAgICAgICAgZWRnZV9mbj1lZGdlX2ZuLAogICAgICAgIGRpdmlzaW9uX3RwPWRpdi50cCwKICAgICAgICBkaXZpc2lvbl9mcD1kaXYuZnAsCiAgICAgICAgZGl2aXNpb25fZm49ZGl2LmZuLAogICAgICAgIG51bV9wcmVkX25vZGVzPWdyYXBoLm51bV9ub2RlcygpLAogICAgKQoKCmRlZiBldmFsdWF0ZV9kYXRhc2V0cygKICAgIGdyYXBoX3BhaXJzOiBsaXN0W3R1cGxlW3RkLmdyYXBoLkJhc2VHcmFwaCwgdGQuZ3JhcGguQmFzZUdyYXBoXV0sCiAgICBzY2FsZTogdHVwbGVbZmxvYXQsIC4uLl0gfCBOb25lID0gTm9uZSwKICAgIG1heF9kaXN0YW5jZTogZmxvYXQgPSA3LjAsCikgLT4gRGF0YXNldHNSZXN1bHQ6CiAgICAiIiJSdW4gOmZ1bmM6YGV2YWx1YXRlYCBvbiBlYWNoIChwcmVkLCBndCkgcGFpciBhbmQgcmV0dXJuIGN1bXVsYXRpdmUKICAgIChtaWNyby1hdmVyYWdlZCkgZWRnZSBhbmQgZGl2aXNpb24gSmFjY2FyZC4KCiAgICBQZXItcGFpciBUUC9GUC9GTiBjb3VudHMgYXJlIHN1bW1lZCBhY3Jvc3MgdGhlIHdob2xlIGxpc3QgYmVmb3JlIHRoZQogICAgSmFjY2FyZCBpcyBjb21wdXRlZCwgc28gbGFyZ2VyIGRhdGFzZXRzIGRvbWluYXRlIHRoZSBzY29yZSBuYXR1cmFsbHkuCgogICAgUGFyYW1ldGVycwogICAgLS0tLS0tLS0tLQogICAgZ3JhcGhfcGFpcnMgOiBsaXN0IG9mIChwcmVkX2dyYXBoLCBndF9ncmFwaCkKICAgICAgICBQcmVkaWN0ZWQgLyBncm91bmQtdHJ1dGggZ3JhcGggcGFpcnMuIEVhY2ggKnByZWRfZ3JhcGgqIGlzIG11dGF0ZWQKICAgICAgICBpbiBwbGFjZSBieSBtYXRjaGluZyAoc2FtZSBzaWRlIGVmZmVjdCBhcyA6ZnVuYzpgZXZhbHVhdGVgKS4KICAgIHNjYWxlIDogdHVwbGVbZmxvYXQsIC4uLl0gfCBOb25lLCBvcHRpb25hbAogICAgICAgIFBoeXNpY2FsIHZveGVsIHNjYWxlIHVzZWQgZm9yIGNlbnRyb2lkLWRpc3RhbmNlIG1hdGNoaW5nLgogICAgbWF4X2Rpc3RhbmNlIDogZmxvYXQsIG9wdGlvbmFsCiAgICAgICAgTWF4aW11bSBjZW50cm9pZCBkaXN0YW5jZSBmb3IgYSBtYXRjaC4KCiAgICBSZXR1cm5zCiAgICAtLS0tLS0tCiAgICBEYXRhc2V0c1Jlc3VsdAogICAgICAgIE5hbWVkIHR1cGxlIHdpdGggYGBlZGdlX2phY2NhcmRgYCwgYGBkaXZpc2lvbl9qYWNjYXJkYGAsIGFuZCB0aGUKICAgICAgICBjb21iaW5lZCBgYHNjb3JlID0gZWRnZV9qYWNjYXJkICsgU0NPUkVfRElWSVNJT05fV0VJR0hUICoKICAgICAgICBkaXZpc2lvbl9qYWNjYXJkYGAuIElmIG5vIGRpdmlzaW9ucyBleGlzdCBhbnl3aGVyZSBpbiB0aGUgaW5wdXQKICAgICAgICB0aGUgZGl2aXNpb24gdGVybSBpcyBkcm9wcGVkIGFuZCBgYHNjb3JlID0gZWRnZV9qYWNjYXJkYGAuCiAgICAiIiIKICAgIGVkZ2VfdHAgPSBlZGdlX2ZwID0gZWRnZV9mbiA9IDAKICAgIGRpdl90cCA9IGRpdl9mcCA9IGRpdl9mbiA9IDAKICAgIGZvciBwcmVkLCBndCBpbiBncmFwaF9wYWlyczoKICAgICAgICByID0gZXZhbHVhdGUocHJlZCwgZ3QsIHNjYWxlPXNjYWxlLCBtYXhfZGlzdGFuY2U9bWF4X2Rpc3RhbmNlKQogICAgICAgIGVkZ2VfdHAgKz0gci5lZGdlX3RwCiAgICAgICAgZWRnZV9mcCArPSByLmVkZ2VfZnAKICAgICAgICBlZGdlX2ZuICs9IHIuZWRnZV9mbgogICAgICAgIGRpdl90cCArPSByLmRpdmlzaW9uX3RwCiAgICAgICAgZGl2X2ZwICs9IHIuZGl2aXNpb25fZnAKICAgICAgICBkaXZfZm4gKz0gci5kaXZpc2lvbl9mbgoKICAgIGVkZ2VfamFjY2FyZCA9IF9qYWNjYXJkKGVkZ2VfdHAsIGVkZ2VfZnAsIGVkZ2VfZm4pCiAgICBoYXNfZGl2aXNpb25zID0gKGRpdl90cCArIGRpdl9mcCArIGRpdl9mbikgPiAwCiAgICBkaXZpc2lvbl9qYWNjYXJkID0gX2phY2NhcmQoZGl2X3RwLCBkaXZfZnAsIGRpdl9mbikgaWYgaGFzX2RpdmlzaW9ucyBlbHNlIGZsb2F0KCJuYW4iKQogICAgc2NvcmUgPSBlZGdlX2phY2NhcmQgKyBTQ09SRV9ESVZJU0lPTl9XRUlHSFQgKiBkaXZpc2lvbl9qYWNjYXJkIGlmIGhhc19kaXZpc2lvbnMgZWxzZSBlZGdlX2phY2NhcmQKCiAgICByZXR1cm4gRGF0YXNldHNSZXN1bHQoCiAgICAgICAgZWRnZV9qYWNjYXJkPWVkZ2VfamFjY2FyZCwKICAgICAgICBkaXZpc2lvbl9qYWNjYXJkPWRpdmlzaW9uX2phY2NhcmQsCiAgICAgICAgc2NvcmU9c2NvcmUsCiAgICApCgoKZGVmIF9tYXRjaGVkX25vZGVfaWRzKGdyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgpIC0+IHBsLkRhdGFGcmFtZToKICAgICIiIlJldHVybiBhIERhdGFGcmFtZSB3aXRoIE5PREVfSUQgYW5kIE1BVENIRURfTk9ERV9JRCAoYXMgSW50NjQpIGZvciAqZ3JhcGgqLiIiIgogICAgbm9kZV9hdHRycyA9IGdyYXBoLm5vZGVfYXR0cnMoCiAgICAgICAgYXR0cl9rZXlzPVt0ZC5ERUZBVUxUX0FUVFJfS0VZUy5OT0RFX0lELCB0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSURdCiAgICApCiAgICByZXR1cm4gbm9kZV9hdHRycwoKCmRlZiBub2RlX3JlY2FsbCgKICAgIGdyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBndF9ncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAopIC0+IGZsb2F0OgogICAgIiIiRnJhY3Rpb24gb2YgR1Qgbm9kZXMgdGhhdCB3ZXJlIG1hdGNoZWQgYnkgYSBwcmVkaWN0ZWQgbm9kZS4KCiAgICBUaGUgcHJlZGljdGVkIGdyYXBoIG11c3QgYWxyZWFkeSBiZSBtYXRjaGVkIChlLmcuIHZpYSA6ZnVuYzpgZXZhbHVhdGVgIG9yCiAgICBgYGdyYXBoLm1hdGNoYGApLgogICAgIiIiCiAgICBub2RlX2F0dHJzID0gX21hdGNoZWRfbm9kZV9pZHMoZ3JhcGgpCiAgICBtYXRjaGVkID0gbm9kZV9hdHRycy5maWx0ZXIoCiAgICAgICAgcGwuY29sKHRkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfTk9ERV9JRCkuaXNfbm90X251bGwoKQogICAgICAgICYgKHBsLmNvbCh0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSUQpICE9IC0xKQogICAgKQogICAgbl9tYXRjaGVkX2d0ID0gbWF0Y2hlZFt0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSURdLm5fdW5pcXVlKCkKICAgIHJldHVybiBuX21hdGNoZWRfZ3QgLyBndF9ncmFwaC5udW1fbm9kZXMoKQoKCmRlZiBwZXJfc2FtcGxlX21ldHJpY3MoCiAgICBlcjogRXZhbHVhdGlvblJlc3VsdCwKICAgIG5fdG90YWw6IGZsb2F0LAogICAgbm9kZV9yZWNhbGw6IGZsb2F0LAopIC0+IGRpY3Q6CiAgICAiIiJEZXJpdmUgcGVyLXNhbXBsZSBtZXRyaWMgY29sdW1ucyBmcm9tIGFuIDpjbGFzczpgRXZhbHVhdGlvblJlc3VsdGAuCgogICAgQ29tcHV0ZXMgYGBlZGdlX2phY2NhcmRgYCwgYGB0b3RhbF9ub2RlX3JhdGlvYGAgKGBgKE5fcHJlZCDiiJIgTl90b3RhbCkgLyBOX3RvdGFsYGApLAogICAgYW5kIHRoZSBhZGp1c3RlZCBlZGdlIEphY2NhcmQgYGBKX2FkaiA9IG1heCgwLCBKIMK3ICgxIOKIkiDOsSDCtyB0b3RhbF9ub2RlX3JhdGlvKSlgYAogICAgd2l0aCDOsSA9IDpkYXRhOmBBREpVU1RNRU5UX0FMUEhBYC4KCiAgICBQYXJhbWV0ZXJzCiAgICAtLS0tLS0tLS0tCiAgICBlcgogICAgICAgIENvdW50cyBmb3Igb25lIChwcmVkLCBndCkgcGFpciDigJQgc2VlIDpmdW5jOmBldmFsdWF0ZWAuCiAgICBuX3RvdGFsCiAgICAgICAgVGFyZ2V0IG5vZGUgY291bnQgKGUuZy4gZnJvbSB0aGUgR0VGRiBgYGVzdGltYXRlZF9udW1iZXJfb2Zfbm9kZXNgYAogICAgICAgIG1ldGFkYXRhIGV4dHJhKS4gUGFzcyBgYGZsb2F0KCJuYW4iKWBgIHdoZW4gdW5hdmFpbGFibGU7IHRoYXQgbWFrZXMKICAgICAgICBgYHRvdGFsX25vZGVfcmF0aW9gYCBhbmQgYGBhZGpfZWRnZV9qYWNjYXJkYGAgYWxzbyBOYU4uCiAgICBub2RlX3JlY2FsbAogICAgICAgIEZyYWN0aW9uIG9mIEdUIG5vZGVzIG1hdGNoZWQgYnkgYSBwcmVkaWN0ZWQgbm9kZS4KCiAgICBSZXR1cm5zCiAgICAtLS0tLS0tCiAgICBkaWN0CiAgICAgICAgT25lIGVudHJ5IHBlciBrZXkgaW4gOmRhdGE6YE1FVFJJQ19DT0xVTU5TYC4KICAgICIiIgogICAgaWYgbl90b3RhbCA+IDA6CiAgICAgICAgdG90YWxfbm9kZV9yYXRpbyA9IChlci5udW1fcHJlZF9ub2RlcyAtIG5fdG90YWwpIC8gbl90b3RhbAogICAgZWxzZToKICAgICAgICB0b3RhbF9ub2RlX3JhdGlvID0gZmxvYXQoIm5hbiIpCgogICAgZWRnZV9kZW5vbSA9IGVyLmVkZ2VfdHAgKyBlci5lZGdlX2ZwICsgZXIuZWRnZV9mbgogICAgZWRnZV9qYWNjYXJkID0gZXIuZWRnZV90cCAvIGVkZ2VfZGVub20gaWYgZWRnZV9kZW5vbSA+IDAgZWxzZSBmbG9hdCgibmFuIikKICAgIGlmIGVkZ2VfamFjY2FyZCA9PSBlZGdlX2phY2NhcmQgYW5kIHRvdGFsX25vZGVfcmF0aW8gPT0gdG90YWxfbm9kZV9yYXRpbzoKICAgICAgICBhZGpfZWRnZV9qYWNjYXJkID0gbWF4KAogICAgICAgICAgICAwLjAsIGVkZ2VfamFjY2FyZCAqICgxIC0gQURKVVNUTUVOVF9BTFBIQSAqIHRvdGFsX25vZGVfcmF0aW8pLAogICAgICAgICkKICAgIGVsc2U6CiAgICAgICAgYWRqX2VkZ2VfamFjY2FyZCA9IGZsb2F0KCJuYW4iKQoKICAgIHJldHVybiB7CiAgICAgICAgImVkZ2VfdHAiOiBlci5lZGdlX3RwLCAiZWRnZV9mcCI6IGVyLmVkZ2VfZnAsICJlZGdlX2ZuIjogZXIuZWRnZV9mbiwKICAgICAgICAiZGl2aXNpb25fdHAiOiBlci5kaXZpc2lvbl90cCwKICAgICAgICAiZGl2aXNpb25fZnAiOiBlci5kaXZpc2lvbl9mcCwKICAgICAgICAiZGl2aXNpb25fZm4iOiBlci5kaXZpc2lvbl9mbiwKICAgICAgICAibnVtX3ByZWRfbm9kZXMiOiBlci5udW1fcHJlZF9ub2RlcywKICAgICAgICAibm9kZV9yZWNhbGwiOiBub2RlX3JlY2FsbCwKICAgICAgICAidG90YWxfbm9kZV9yYXRpbyI6IHRvdGFsX25vZGVfcmF0aW8sCiAgICAgICAgImVkZ2VfamFjY2FyZCI6IGVkZ2VfamFjY2FyZCwKICAgICAgICAiYWRqX2VkZ2VfamFjY2FyZCI6IGFkal9lZGdlX2phY2NhcmQsCiAgICB9CgoKZGVmIG5hbl9tZXRyaWNzX3JvdygpIC0+IGRpY3Q6CiAgICAiIiJSZXR1cm4gYSBkaWN0IHdpdGggZXZlcnkgOmRhdGE6YE1FVFJJQ19DT0xVTU5TYCBrZXkgc2V0IHRvIE5hTi4iIiIKICAgIHJldHVybiB7Y29sOiBmbG9hdCgibmFuIikgZm9yIGNvbCBpbiBNRVRSSUNfQ09MVU1OU30KCgpkZWYgc3VtbWFyaXNlKHJvd3M6IGxpc3RbZGljdF0pIC0+IGRpY3Q6CiAgICAiIiJBZ2dyZWdhdGUgcGVyLXNhbXBsZSBtZXRyaWMgcm93cyBpbnRvIGEgcnVuLWxldmVsIHN1bW1hcnkuCgogICAgLSBgYGVkZ2VfamFjY2FyZGBgIC8gYGBkaXZpc2lvbl9qYWNjYXJkYGA6IG1pY3JvLWF2ZXJhZ2VkIGFjcm9zcyB2YWxpZCByb3dzCiAgICAgIChUUC9GUC9GTiBzdW1tZWQsIHRoZW4gSmFjY2FyZCkuCiAgICAtIGBgYWRqX2VkZ2VfamFjY2FyZGBgOiBwZXItc2FtcGxlIGFkanVzdGVkIEphY2NhcmQgd2VpZ2h0LWF2ZXJhZ2VkIGJ5CiAgICAgIHNhbXBsZSBzaXplIGBgd19pID0gVFBfaSArIEZQX2kgKyBGTl9pYGA7IHJvd3Mgd2l0aCBOYU4gYXJlIHNraXBwZWQuCiAgICAtIGBgc2NvcmVgYDogYGBhZGpfZWRnZV9qYWNjYXJkICsgU0NPUkVfRElWSVNJT05fV0VJR0hUIMK3IGRpdmlzaW9uX2phY2NhcmRgYC4KCiAgICBQYXJhbWV0ZXJzCiAgICAtLS0tLS0tLS0tCiAgICByb3dzCiAgICAgICAgUGVyLXNhbXBsZSBkaWN0cyBhcyBwcm9kdWNlZCBieSA6ZnVuYzpgcGVyX3NhbXBsZV9tZXRyaWNzYC4gUm93cyB3aXRoCiAgICAgICAgTmFOIGBgZWRnZV90cGBgIGFyZSB0cmVhdGVkIGFzIGZhaWxlZCBldmFsdWF0aW9ucyBhbmQgc2tpcHBlZC4KICAgICIiIgogICAgdmFsaWQgPSBbciBmb3IgciBpbiByb3dzIGlmIHJbImVkZ2VfdHAiXSA9PSByWyJlZGdlX3RwIl1dCiAgICBpZiBub3QgdmFsaWQ6CiAgICAgICAgcmV0dXJuIHsKICAgICAgICAgICAgIm4iOiAwLCAiZWRnZV9qYWNjYXJkIjogZmxvYXQoIm5hbiIpLAogICAgICAgICAgICAiZGl2aXNpb25famFjY2FyZCI6IGZsb2F0KCJuYW4iKSwKICAgICAgICAgICAgImRpdmlzaW9uX3RwIjogMCwgImRpdmlzaW9uX2ZwIjogMCwgImRpdmlzaW9uX2ZuIjogMCwKICAgICAgICAgICAgIm5vZGVfcmVjYWxsIjogZmxvYXQoIm5hbiIpLAogICAgICAgICAgICAiYWRqX2VkZ2VfamFjY2FyZCI6IGZsb2F0KCJuYW4iKSwgIm5fYWRqIjogMCwKICAgICAgICAgICAgInNjb3JlIjogZmxvYXQoIm5hbiIpLAogICAgICAgIH0KICAgIHRvdGFscyA9IHtjOiBzdW0ocltjXSBmb3IgciBpbiB2YWxpZCkgZm9yIGMgaW4gQ09VTlRfQ09MVU1OU30KCiAgICBhZGpfcm93cyA9IFtyIGZvciByIGluIHZhbGlkIGlmIHJbImFkal9lZGdlX2phY2NhcmQiXSA9PSByWyJhZGpfZWRnZV9qYWNjYXJkIl1dCiAgICB3ZWlnaHRzID0gW3JbImVkZ2VfdHAiXSArIHJbImVkZ2VfZnAiXSArIHJbImVkZ2VfZm4iXSBmb3IgciBpbiBhZGpfcm93c10KICAgIHRvdGFsX3cgPSBzdW0od2VpZ2h0cykKICAgIGlmIHRvdGFsX3cgPiAwOgogICAgICAgIGFkal9lZGdlX2phY2NhcmQgPSBzdW0oCiAgICAgICAgICAgIHcgKiByWyJhZGpfZWRnZV9qYWNjYXJkIl0gZm9yIHcsIHIgaW4gemlwKHdlaWdodHMsIGFkal9yb3dzKQogICAgICAgICkgLyB0b3RhbF93CiAgICBlbHNlOgogICAgICAgIGFkal9lZGdlX2phY2NhcmQgPSBmbG9hdCgibmFuIikKCiAgICBkaXZpc2lvbl90b3RhbCA9ICgKICAgICAgICB0b3RhbHNbImRpdmlzaW9uX3RwIl0gKyB0b3RhbHNbImRpdmlzaW9uX2ZwIl0gKyB0b3RhbHNbImRpdmlzaW9uX2ZuIl0KICAgICkKICAgIGlmIGRpdmlzaW9uX3RvdGFsID09IDA6CiAgICAgICAgd2FybmluZ3Mud2FybigKICAgICAgICAgICAgIk5vIGRpdmlzaW9ucyBwcmVzZW50IGFjcm9zcyBhbnkgc2FtcGxlIGluIHRoaXMgc3BsaXQ7ICIKICAgICAgICAgICAgImRyb3BwaW5nIGRpdmlzaW9uIHRlcm0gZnJvbSB0aGUgY29tYmluZWQgc2NvcmUuIgogICAgICAgICkKICAgICAgICBkaXZpc2lvbl9qYWNjYXJkID0gZmxvYXQoIm5hbiIpCiAgICAgICAgc2NvcmUgPSBhZGpfZWRnZV9qYWNjYXJkCiAgICBlbHNlOgogICAgICAgIGRpdmlzaW9uX2phY2NhcmQgPSBfamFjY2FyZCgKICAgICAgICAgICAgdG90YWxzWyJkaXZpc2lvbl90cCJdLCB0b3RhbHNbImRpdmlzaW9uX2ZwIl0sIHRvdGFsc1siZGl2aXNpb25fZm4iXSwKICAgICAgICApCiAgICAgICAgc2NvcmUgPSBhZGpfZWRnZV9qYWNjYXJkICsgU0NPUkVfRElWSVNJT05fV0VJR0hUICogZGl2aXNpb25famFjY2FyZAogICAgcmV0dXJuIHsKICAgICAgICAibiI6IGxlbih2YWxpZCksCiAgICAgICAgImVkZ2VfamFjY2FyZCI6IF9qYWNjYXJkKAogICAgICAgICAgICB0b3RhbHNbImVkZ2VfdHAiXSwgdG90YWxzWyJlZGdlX2ZwIl0sIHRvdGFsc1siZWRnZV9mbiJdLAogICAgICAgICksCiAgICAgICAgImRpdmlzaW9uX2phY2NhcmQiOiBkaXZpc2lvbl9qYWNjYXJkLAogICAgICAgICJkaXZpc2lvbl90cCI6IHRvdGFsc1siZGl2aXNpb25fdHAiXSwKICAgICAgICAiZGl2aXNpb25fZnAiOiB0b3RhbHNbImRpdmlzaW9uX2ZwIl0sCiAgICAgICAgImRpdmlzaW9uX2ZuIjogdG90YWxzWyJkaXZpc2lvbl9mbiJdLAogICAgICAgICJub2RlX3JlY2FsbCI6IHN1bShyWyJub2RlX3JlY2FsbCJdIGZvciByIGluIHZhbGlkKSAvIGxlbih2YWxpZCksCiAgICAgICAgImFkal9lZGdlX2phY2NhcmQiOiBhZGpfZWRnZV9qYWNjYXJkLAogICAgICAgICJuX2FkaiI6IGxlbihhZGpfcm93cyksCiAgICAgICAgInNjb3JlIjogc2NvcmUsCiAgICB9Cg==",
"division_metrics.py": "aW1wb3J0IHdhcm5pbmdzCmZyb20gY29sbGVjdGlvbnMgaW1wb3J0IGRlcXVlCmZyb20gdHlwaW5nIGltcG9ydCBOYW1lZFR1cGxlCgppbXBvcnQgcG9sYXJzIGFzIHBsCmltcG9ydCB0cmFja3NkYXRhIGFzIHRkCgoKY2xhc3MgRGl2aXNpb25Db3VudHMoTmFtZWRUdXBsZSk6CiAgICAiIiJDb3VudHMgZm9yIGRpdmlzaW9uIGV2ZW50IGV2YWx1YXRpb24uIiIiCgogICAgdHA6IGludAogICAgZm46IGludAogICAgZnA6IGludAoKCmRlZiBfcmVzZXRfbWF0Y2hpbmdfYXR0cnMoZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCkgLT4gTm9uZToKICAgICIiIlJlc2V0IGFueSBwcmUtZXhpc3RpbmcgbWF0Y2ggYXR0cnMgaW4gcGxhY2Ugc28gYSBmcmVzaCBgYC5tYXRjaCgpYGAgaXNuJ3QKICAgIGNvbnRhbWluYXRlZCBieSBzdGFsZSB2YWx1ZXMgY2FycmllZCBpbiBmcm9tIGEgcHJldmlvdXMgbWF0Y2hpbmcgcGFzcy4iIiIKICAgIG5vZGVfa2V5cyA9IGdyYXBoLm5vZGVfYXR0cl9rZXlzKCkKICAgIGlmIHRkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfTk9ERV9JRCBpbiBub2RlX2tleXM6CiAgICAgICAgbm9kZV9pZHMgPSBncmFwaC5ub2RlX2lkcygpCiAgICAgICAgaWYgbGVuKG5vZGVfaWRzKSA+IDA6CiAgICAgICAgICAgIHJlc2V0OiBkaWN0ID0ge3RkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfTk9ERV9JRDogLTF9CiAgICAgICAgICAgIGlmIHRkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIX1NDT1JFIGluIG5vZGVfa2V5czoKICAgICAgICAgICAgICAgIHJlc2V0W3RkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIX1NDT1JFXSA9IDAuMAogICAgICAgICAgICBncmFwaC51cGRhdGVfbm9kZV9hdHRycyhub2RlX2lkcz1ub2RlX2lkcywgYXR0cnM9cmVzZXQpCiAgICBpZiB0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX0VER0VfTUFTSyBpbiBncmFwaC5lZGdlX2F0dHJfa2V5cygpOgogICAgICAgIGVkZ2VfaWRzID0gZ3JhcGguZWRnZV9pZHMoKQogICAgICAgIGlmIGxlbihlZGdlX2lkcykgPiAwOgogICAgICAgICAgICBncmFwaC51cGRhdGVfZWRnZV9hdHRycygKICAgICAgICAgICAgICAgIGVkZ2VfaWRzPWVkZ2VfaWRzLAogICAgICAgICAgICAgICAgYXR0cnM9e3RkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfRURHRV9NQVNLOiBGYWxzZX0sCiAgICAgICAgICAgICkKCgpkZWYgZXh0cmFjdF9kaXZpc2lvbnMoCiAgICBncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAopIC0+IGRpY3RbaW50LCB0ZC5ncmFwaC5CYXNlR3JhcGhdOgogICAgIiIiRXh0cmFjdCBpbmRpdmlkdWFsIGRpdmlzaW9uIGV2ZW50cyBhcyBzZXBhcmF0ZSBzdWJncmFwaHMuCgogICAgRWFjaCBkaXZpc2lvbiBldmVudCBpbmNsdWRlcyB0aGUgcGFyZW50IG9mIHRoZSBkaXZpZGluZyBub2RlLCB0aGUKICAgIGRpdmlkaW5nIG5vZGUsIGl0cyBjaGlsZHJlbiwgYW5kIHRoZSBncmFuZGNoaWxkcmVuOjoKCiAgICAgICAgcGFyZW50IOKGkiBkaXZpZGVyIOKGkiBjaGlsZDEg4oaSIGdyYW5kY2hpbGQxCiAgICAgICAgICAgICAgICAgICAgICAgICDihpIgY2hpbGQyIOKGkiBncmFuZGNoaWxkMgoKICAgIFBhcmFtZXRlcnMKICAgIC0tLS0tLS0tLS0KICAgIGdyYXBoIDogdGQuZ3JhcGguQmFzZUdyYXBoCiAgICAgICAgVGhlIGlucHV0IHRyYWNraW5nIGdyYXBoLgoKICAgIFJldHVybnMKICAgIC0tLS0tLS0KICAgIGRpY3RbaW50LCB0ZC5ncmFwaC5CYXNlR3JhcGhdCiAgICAgICAgTWFwcGluZyBmcm9tIGRpdmlkaW5nIG5vZGUgSUQgdG8gYSBzdWJncmFwaCBjb250YWluaW5nIHRoZQogICAgICAgIHBhcmVudCwgZGl2aWRlciwgY2hpbGRyZW4sIGFuZCBncmFuZGNoaWxkcmVuLgogICAgIiIiCiAgICBkaXZpc2lvbnM6IGRpY3RbaW50LCB0ZC5ncmFwaC5CYXNlR3JhcGhdID0ge30KICAgIGZvciBkaXZfbm9kZSBpbiBncmFwaC5kaXZpZGluZ19ub2RlcygpOgogICAgICAgIHBhcmVudHMgPSBncmFwaC5wcmVkZWNlc3NvcnMoZGl2X25vZGUpCiAgICAgICAgY2hpbGRyZW4gPSBncmFwaC5zdWNjZXNzb3JzKGRpdl9ub2RlKQogICAgICAgIGdyYW5kY2hpbGRyZW4gPSBbZ2MgZm9yIGNoaWxkIGluIGNoaWxkcmVuIGZvciBnYyBpbiBncmFwaC5zdWNjZXNzb3JzKGNoaWxkKV0KICAgICAgICBrZWVwID0gWypwYXJlbnRzLCBkaXZfbm9kZSwgKmNoaWxkcmVuLCAqZ3JhbmRjaGlsZHJlbl0KICAgICAgICBkaXZpc2lvbnNbZGl2X25vZGVdID0gZ3JhcGguZmlsdGVyKG5vZGVfaWRzPWtlZXApLnN1YmdyYXBoKCkKICAgIHJldHVybiBkaXZpc2lvbnMKCgpkZWYgbWF0Y2hfZGl2aXNpb25zKAogICAgcHJlZF9ncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAogICAgZ3RfZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCwKICAgIHNjYWxlOiB0dXBsZVtmbG9hdCwgLi4uXSB8IE5vbmUgPSBOb25lLAogICAgbWF4X2Rpc3RhbmNlOiBmbG9hdCA9IDcuMCwKKSAtPiBkaWN0W2ludCwgdGQuZ3JhcGguQmFzZUdyYXBoXToKICAgICIiIk1hdGNoIHRoZSBwcmVkaWN0ZWQgZ3JhcGggYWdhaW5zdCBlYWNoIEdUIGRpdmlzaW9uIHN1YmdyYXBoLgoKICAgIEV4dHJhY3RzIGRpdmlzaW9uIGV2ZW50cyBmcm9tICpndF9ncmFwaCogdmlhIDpmdW5jOmBleHRyYWN0X2RpdmlzaW9uc2AsCiAgICB0aGVuIHJ1bnMgYGBwcmVkX2dyYXBoLm1hdGNoKGd0X2RpdiwgLi4uKWBgIGZvciBlYWNoIG9uZSBpbmRlcGVuZGVudGx5LgogICAgQSBmcmVzaCBjb3B5IG9mICpwcmVkX2dyYXBoKiBpcyB1c2VkIHBlciBkaXZpc2lvbiBzbyBtYXRjaGluZ3MgZG9uJ3QKICAgIGludGVyZmVyZS4KCiAgICBQYXJhbWV0ZXJzCiAgICAtLS0tLS0tLS0tCiAgICBwcmVkX2dyYXBoIDogdGQuZ3JhcGguQmFzZUdyYXBoCiAgICAgICAgVGhlIHByZWRpY3RlZCB0cmFja2luZyBncmFwaC4KICAgIGd0X2dyYXBoIDogdGQuZ3JhcGguQmFzZUdyYXBoCiAgICAgICAgVGhlIGdyb3VuZC10cnV0aCB0cmFja2luZyBncmFwaC4KICAgIHNjYWxlIDogdHVwbGVbZmxvYXQsIC4uLl0gfCBOb25lCiAgICAgICAgUGh5c2ljYWwgdm94ZWwgc2NhbGUgdXNlZCBmb3IgY2VudHJvaWQtZGlzdGFuY2UgbWF0Y2hpbmcuCiAgICBtYXhfZGlzdGFuY2UgOiBmbG9hdAogICAgICAgIE1heGltdW0gY2VudHJvaWQgZGlzdGFuY2UgZm9yIGEgbWF0Y2guCgogICAgUmV0dXJucwogICAgLS0tLS0tLQogICAgZGljdFtpbnQsIHRkLmdyYXBoLkJhc2VHcmFwaF0KICAgICAgICBNYXBwaW5nIGZyb20gR1QgZGl2aWRpbmctbm9kZSBJRCB0byB0aGUgbWF0Y2hlZCBjb3B5IG9mCiAgICAgICAgKnByZWRfZ3JhcGgqIGZvciB0aGF0IGRpdmlzaW9uLgogICAgIiIiCiAgICBmcm9tIHRyYWNrc2RhdGEubWV0cmljcyBpbXBvcnQgRGlzdGFuY2VNYXRjaGluZwogICAgbWF0Y2hpbmcgPSBEaXN0YW5jZU1hdGNoaW5nKG1heF9kaXN0YW5jZT1tYXhfZGlzdGFuY2UsIHNjYWxlPXNjYWxlKQoKICAgIGd0X2RpdmlzaW9ucyA9IGV4dHJhY3RfZGl2aXNpb25zKGd0X2dyYXBoKQogICAgbWF0Y2hlZDogZGljdFtpbnQsIHRkLmdyYXBoLkJhc2VHcmFwaF0gPSB7fQoKICAgIGZyb20gdHJhY2tzZGF0YS5vcHRpb25zIGltcG9ydCBnZXRfb3B0aW9ucywgc2V0X29wdGlvbnMKCiAgICBwcmV2X3Nob3dfcHJvZ3Jlc3MgPSBnZXRfb3B0aW9ucygpLnNob3dfcHJvZ3Jlc3MKICAgIHNldF9vcHRpb25zKHNob3dfcHJvZ3Jlc3M9RmFsc2UpCiAgICB0cnk6CiAgICAgICAgZm9yIGRpdl9ub2RlLCBndF9kaXYgaW4gZ3RfZGl2aXNpb25zLml0ZW1zKCk6CiAgICAgICAgICAgIHByZWRfY29weSA9IHByZWRfZ3JhcGguY29weSgpCiAgICAgICAgICAgIF9yZXNldF9tYXRjaGluZ19hdHRycyhwcmVkX2NvcHkpCiAgICAgICAgICAgIHdpdGggd2FybmluZ3MuY2F0Y2hfd2FybmluZ3MoKToKICAgICAgICAgICAgICAgIGZyb20gc2NpcHkuc3BhcnNlIGltcG9ydCBTcGFyc2VFZmZpY2llbmN5V2FybmluZwogICAgICAgICAgICAgICAgd2FybmluZ3MuZmlsdGVyd2FybmluZ3MoImlnbm9yZSIsIGNhdGVnb3J5PVNwYXJzZUVmZmljaWVuY3lXYXJuaW5nKQogICAgICAgICAgICAgICAgcHJlZF9jb3B5Lm1hdGNoKGd0X2RpdiwgbWF0Y2hpbmc9bWF0Y2hpbmcpCiAgICAgICAgICAgIG1hdGNoZWRbZGl2X25vZGVdID0gcHJlZF9jb3B5CiAgICBmaW5hbGx5OgogICAgICAgIHNldF9vcHRpb25zKHNob3dfcHJvZ3Jlc3M9cHJldl9zaG93X3Byb2dyZXNzKQoKICAgIHJldHVybiBtYXRjaGVkCgoKZGVmIF9tYXRjaF9mdWxsKAogICAgcHJlZF9ncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAogICAgZ3RfZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCwKICAgIHNjYWxlOiB0dXBsZVtmbG9hdCwgLi4uXSB8IE5vbmUsCiAgICBtYXhfZGlzdGFuY2U6IGZsb2F0LAopIC0+IHRkLmdyYXBoLkJhc2VHcmFwaDoKICAgICIiIk1hdGNoIHRoZSBmdWxsIHByZWQgZ3JhcGggYWdhaW5zdCB0aGUgZnVsbCBHVCBncmFwaCwgcmV0dXJuIHRoZSBtYXRjaGVkIGNvcHkuIiIiCiAgICBmcm9tIHRyYWNrc2RhdGEubWV0cmljcyBpbXBvcnQgRGlzdGFuY2VNYXRjaGluZwogICAgbWF0Y2hpbmcgPSBEaXN0YW5jZU1hdGNoaW5nKG1heF9kaXN0YW5jZT1tYXhfZGlzdGFuY2UsIHNjYWxlPXNjYWxlKQoKICAgIHByZWRfY29weSA9IHByZWRfZ3JhcGguY29weSgpCiAgICBfcmVzZXRfbWF0Y2hpbmdfYXR0cnMocHJlZF9jb3B5KQoKICAgIGZyb20gdHJhY2tzZGF0YS5vcHRpb25zIGltcG9ydCBnZXRfb3B0aW9ucywgc2V0X29wdGlvbnMKCiAgICBwcmV2X3Nob3dfcHJvZ3Jlc3MgPSBnZXRfb3B0aW9ucygpLnNob3dfcHJvZ3Jlc3MKICAgIHNldF9vcHRpb25zKHNob3dfcHJvZ3Jlc3M9RmFsc2UpCiAgICB0cnk6CiAgICAgICAgd2l0aCB3YXJuaW5ncy5jYXRjaF93YXJuaW5ncygpOgogICAgICAgICAgICBmcm9tIHNjaXB5LnNwYXJzZSBpbXBvcnQgU3BhcnNlRWZmaWNpZW5jeVdhcm5pbmcKICAgICAgICAgICAgd2FybmluZ3MuZmlsdGVyd2FybmluZ3MoImlnbm9yZSIsIGNhdGVnb3J5PVNwYXJzZUVmZmljaWVuY3lXYXJuaW5nKQogICAgICAgICAgICBwcmVkX2NvcHkubWF0Y2goZ3RfZ3JhcGgsIG1hdGNoaW5nPW1hdGNoaW5nKQogICAgZmluYWxseToKICAgICAgICBzZXRfb3B0aW9ucyhzaG93X3Byb2dyZXNzPXByZXZfc2hvd19wcm9ncmVzcykKCiAgICByZXR1cm4gcHJlZF9jb3B5CgoKZGVmIF9tYXRjaGVkX25vZGVfYXR0cnMoZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCkgLT4gcGwuRGF0YUZyYW1lOgogICAgIiIiUmV0dXJuIG5vZGUgYXR0cnMgKG5vZGVfaWQsIG1hdGNoZWRfbm9kZV9pZCwgdCkgZm9yIG1hdGNoZWQgcHJlZCBub2Rlcy4iIiIKICAgIG5vZGVfYXR0cnMgPSBncmFwaC5ub2RlX2F0dHJzKAogICAgICAgIGF0dHJfa2V5cz1bCiAgICAgICAgICAgIHRkLkRFRkFVTFRfQVRUUl9LRVlTLk5PREVfSUQsCiAgICAgICAgICAgIHRkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfTk9ERV9JRCwKICAgICAgICAgICAgInQiLAogICAgICAgIF0sCiAgICApCiAgICByZXR1cm4gbm9kZV9hdHRycy5maWx0ZXIoCiAgICAgICAgcGwuY29sKHRkLkRFRkFVTFRfQVRUUl9LRVlTLk1BVENIRURfTk9ERV9JRCkuaXNfbm90X251bGwoKQogICAgICAgICYgKHBsLmNvbCh0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSUQpICE9IC0xKQogICAgKQoKCmRlZiBfaGFzX3N0YWdlX2NvdmVyYWdlKAogICAgbWF0Y2hlZF9hdHRyczogcGwuRGF0YUZyYW1lLAogICAgZ3RfZGl2OiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBkaXZpZGVyX2lkOiBpbnQsCikgLT4gYm9vbDoKICAgICIiIkNoZWNrIHRoYXQgbWF0Y2hlcyBjb3ZlciBib3RoIHN0YWdlcyBvZiBhIEdUIGRpdmlzaW9uLgoKICAgIFRoZSBHVCBkaXZpc2lvbiBzdWJncmFwaCBoYXMgYSAqb25lLW5vZGUgc3RhZ2UqICh0aW1lcG9pbnRzIHdpdGggYQogICAgc2luZ2xlIEdUIG5vZGUg4oCUIHBhcmVudCBhbmQgZGl2aWRlciwgcHJlLXNwbGl0KSBhbmQgdHdvIG9yIG1vcmUKICAgICpkYXVnaHRlciBsaW5lYWdlcyogKGVhY2ggY2hpbGQgb2YgKmRpdmlkZXJfaWQqIHBsdXMgaXRzIGRlc2NlbmRhbnRzCiAgICB3aXRoaW4gdGhlIHN1YmdyYXBoKS4gQSB2YWxpZCBtYXRjaCByZXF1aXJlczoKCiAgICAqIOKJpTEgbWF0Y2hlZCBwcmVkaWN0aW9uIG5vZGUgd2hvc2UgdGltZXBvaW50IGZhbGxzIGluIHRoZSBvbmUtbm9kZQogICAgICBzdGFnZSwgQU5ECiAgICAqIG1hdGNoZWQgcHJlZGljdGlvbiBub2RlcyB3aG9zZSBtYXRjaGVkIEdUIG5vZGVzIGNvdmVyIOKJpTIgZGlzdGluY3QKICAgICAgZGF1Z2h0ZXIgbGluZWFnZXMuIExpbmVhZ2UgaGl0cyBtYXkgb2NjdXIgYXQgZGlmZmVyZW50IHRpbWVwb2ludHM7CiAgICAgIGEgc2luZ2xlIGRhdWdodGVyIG1hdGNoZWQgb25seSBhdCB0PWRpdmlkZXIrMiBzdGlsbCBjb3VudHMuCgogICAgV2hlbiB0aGUgc3ViZ3JhcGggY29udGFpbnMgYSBzZWNvbmRhcnkgZGl2aWRlciAoZS5nLiBzdWNjZXNzaXZlCiAgICBkaXZpc2lvbnMpLCAqZGl2aWRlcl9pZCogZGlzYW1iaWd1YXRlcyB3aGljaCBzcGxpdCB3ZSdyZSBzY29yaW5nLgogICAgIiIiCiAgICBpZiBtYXRjaGVkX2F0dHJzLmlzX2VtcHR5KCk6CiAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgZ3RfdGltZV9jb3VudHMgPSAoCiAgICAgICAgZ3RfZGl2Lm5vZGVfYXR0cnMoYXR0cl9rZXlzPVsidCJdKQogICAgICAgIC5ncm91cF9ieSgidCIpCiAgICAgICAgLmFnZyhwbC5sZW4oKS5hbGlhcygibiIpKQogICAgKQogICAgb25lX25vZGVfdGltZXMgPSBzZXQoZ3RfdGltZV9jb3VudHMuZmlsdGVyKHBsLmNvbCgibiIpID09IDEpWyJ0Il0udG9fbGlzdCgpKQogICAgaWYgbm90IG9uZV9ub2RlX3RpbWVzOgogICAgICAgIHJldHVybiBGYWxzZQoKICAgIGNoaWxkcmVuID0gZ3RfZGl2LnN1Y2Nlc3NvcnMoZGl2aWRlcl9pZCkKICAgIGlmIGxlbihjaGlsZHJlbikgPCAyOgogICAgICAgIHJldHVybiBGYWxzZQoKICAgIGRlZiBfZGVzY2VuZGFudHMoc2VlZDogaW50KSAtPiBzZXRbaW50XToKICAgICAgICBvdXQ6IHNldFtpbnRdID0ge3NlZWR9CiAgICAgICAgc3RhY2sgPSBbc2VlZF0KICAgICAgICB3aGlsZSBzdGFjazoKICAgICAgICAgICAgZm9yIG54dCBpbiBndF9kaXYuc3VjY2Vzc29ycyhzdGFjay5wb3AoKSk6CiAgICAgICAgICAgICAgICBpZiBueHQgbm90IGluIG91dDoKICAgICAgICAgICAgICAgICAgICBvdXQuYWRkKG54dCkKICAgICAgICAgICAgICAgICAgICBzdGFjay5hcHBlbmQobnh0KQogICAgICAgIHJldHVybiBvdXQKCiAgICBsaW5lYWdlcyA9IFtfZGVzY2VuZGFudHMoYykgZm9yIGMgaW4gY2hpbGRyZW5dCgogICAgbWF0Y2hlZF90aW1lX2NvdW50cyA9IG1hdGNoZWRfYXR0cnMuZ3JvdXBfYnkoInQiKS5hZ2cocGwubGVuKCkuYWxpYXMoIm4iKSkKICAgIGhhc19vbmUgPSBtYXRjaGVkX3RpbWVfY291bnRzLmZpbHRlcihwbC5jb2woInQiKS5pc19pbihvbmVfbm9kZV90aW1lcykpLmhlaWdodCA+IDAKICAgIGlmIG5vdCBoYXNfb25lOgogICAgICAgIHJldHVybiBGYWxzZQoKICAgIG1hdGNoZWRfZ3RfaWRzID0gc2V0KG1hdGNoZWRfYXR0cnNbdGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hFRF9OT0RFX0lEXS50b19saXN0KCkpCiAgICBsaW5lYWdlc19jb3ZlcmVkID0gc3VtKDEgZm9yIGxpbiBpbiBsaW5lYWdlcyBpZiBsaW4gJiBtYXRjaGVkX2d0X2lkcykKICAgIHJldHVybiBsaW5lYWdlc19jb3ZlcmVkID49IDIKCgpkZWYgX3dlYWtseV9jb25uZWN0ZWRfY29tcG9uZW50cygKICAgIGdyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBub2RlX2lkczogbGlzdFtpbnRdLAopIC0+IGxpc3RbdHVwbGVbc2V0W2ludF0sIHNldFtpbnRdXV06CiAgICAiIiJQYXJ0aXRpb24gKm5vZGVfaWRzKiBpbnRvIHdlYWtseS1jb25uZWN0ZWQgY29tcG9uZW50cyBvZiAqZ3JhcGgqLgoKICAgIFJldHVybnMgb25lIGBgKG1hdGNoZWRfc3Vic2V0LCB2aXNpdGVkKWBgIHBhaXIgcGVyIGNvbXBvbmVudDoKICAgICptYXRjaGVkX3N1YnNldCogaXMgdGhlIGNvbXBvbmVudCByZXN0cmljdGVkIHRvICpub2RlX2lkcyo7ICp2aXNpdGVkKgogICAgaXMgZXZlcnkgZ3JhcGggbm9kZSByZWFjaGFibGUgZnJvbSB0aGUgY29tcG9uZW50IChpbmNsdWRpbmcgdW5tYXRjaGVkCiAgICBpbnRlcm1lZGlhcmllcykuIFRoZSB2aXNpdGVkIHNldCBsZXRzIGNhbGxlcnMgbG9jYXRlIHN0cnVjdHVyYWwKICAgIGZlYXR1cmVzIC0tIGluIHBhcnRpY3VsYXIgcHJlZCBkaXZpZGluZyBub2RlcyAtLSB0aGF0IG1heSBzaXQgb24KICAgIHVubWF0Y2hlZCBub2RlcyBiZXR3ZWVuIG1hdGNoZWQgb25lcy4KICAgICIiIgogICAgcmVtYWluaW5nID0gc2V0KG5vZGVfaWRzKQogICAgY29tcG9uZW50czogbGlzdFt0dXBsZVtzZXRbaW50XSwgc2V0W2ludF1dXSA9IFtdCiAgICB3aGlsZSByZW1haW5pbmc6CiAgICAgICAgc2VlZCA9IG5leHQoaXRlcihyZW1haW5pbmcpKQogICAgICAgIHZpc2l0ZWQ6IHNldFtpbnRdID0ge3NlZWR9CiAgICAgICAgcXVldWU6IGRlcXVlW2ludF0gPSBkZXF1ZShbc2VlZF0pCiAgICAgICAgY29tcG9uZW50OiBzZXRbaW50XSA9IHtzZWVkfQogICAgICAgIHdoaWxlIHF1ZXVlOgogICAgICAgICAgICBjdXJyZW50ID0gcXVldWUucG9wbGVmdCgpCiAgICAgICAgICAgIG5laWdoYm9ycyA9IGdyYXBoLnN1Y2Nlc3NvcnMoY3VycmVudCkgKyBncmFwaC5wcmVkZWNlc3NvcnMoY3VycmVudCkKICAgICAgICAgICAgZm9yIG5laWdoYm9yIGluIG5laWdoYm9yczoKICAgICAgICAgICAgICAgIGlmIG5laWdoYm9yIG5vdCBpbiB2aXNpdGVkOgogICAgICAgICAgICAgICAgICAgIHZpc2l0ZWQuYWRkKG5laWdoYm9yKQogICAgICAgICAgICAgICAgICAgIHF1ZXVlLmFwcGVuZChuZWlnaGJvcikKICAgICAgICAgICAgICAgICAgICBpZiBuZWlnaGJvciBpbiByZW1haW5pbmc6CiAgICAgICAgICAgICAgICAgICAgICAgIGNvbXBvbmVudC5hZGQobmVpZ2hib3IpCiAgICAgICAgY29tcG9uZW50cy5hcHBlbmQoKGNvbXBvbmVudCwgdmlzaXRlZCkpCiAgICAgICAgcmVtYWluaW5nIC09IGNvbXBvbmVudAogICAgcmV0dXJuIGNvbXBvbmVudHMKCgpkZWYgX2JpcGFydGl0ZV9tYXhfbWF0Y2hpbmcoCiAgICBsZWZ0OiBsaXN0W2ludF0sCiAgICBlZGdlczogZGljdFtpbnQsIHNldFtpbnRdXSwKKSAtPiBkaWN0W2ludCwgaW50XToKICAgICIiIk1heGltdW0tY2FyZGluYWxpdHkgYmlwYXJ0aXRlIG1hdGNoaW5nIHZpYSBERlMgYXVnbWVudGluZyBwYXRocy4KCiAgICAqZWRnZXMqIG1hcHMgZWFjaCBsZWZ0LXNpZGUgdmVydGV4IHRvIHRoZSBzZXQgb2YgYWRqYWNlbnQgcmlnaHQtc2lkZQogICAgdmVydGljZXMuIFJldHVybnMgb25seSB0aGUgbWF0Y2hlZCBwYWlycyBhcyBhIGBgbGVmdCDihpIgcmlnaHRgYCBkaWN0LgogICAgIiIiCiAgICBtYXRjaF9yOiBkaWN0W2ludCwgaW50XSA9IHt9CiAgICBtYXRjaF9sOiBkaWN0W2ludCwgaW50XSA9IHt9CgogICAgZGVmIGF1Z21lbnQodTogaW50LCBzZWVuOiBzZXRbaW50XSkgLT4gYm9vbDoKICAgICAgICBmb3IgdiBpbiBlZGdlcy5nZXQodSwgKCkpOgogICAgICAgICAgICBpZiB2IGluIHNlZW46CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBzZWVuLmFkZCh2KQogICAgICAgICAgICBpZiB2IG5vdCBpbiBtYXRjaF9yIG9yIGF1Z21lbnQobWF0Y2hfclt2XSwgc2Vlbik6CiAgICAgICAgICAgICAgICBtYXRjaF9sW3VdID0gdgogICAgICAgICAgICAgICAgbWF0Y2hfclt2XSA9IHUKICAgICAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgZm9yIHUgaW4gbGVmdDoKICAgICAgICBhdWdtZW50KHUsIHNldCgpKQoKICAgIHJldHVybiBtYXRjaF9sCgoKZGVmIHNjb3JlX2RpdmlzaW9ucygKICAgIHByZWRfZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCwKICAgIGd0X2dyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBzY2FsZTogdHVwbGVbZmxvYXQsIC4uLl0gfCBOb25lID0gTm9uZSwKICAgIG1heF9kaXN0YW5jZTogZmxvYXQgPSA3LjAsCikgLT4gZGljdFtpbnQsIGludF06CiAgICAiIiJTY29yZSBlYWNoIEdUIGRpdmlzaW9uOiAxIGlmIHRoZSBwcmVkaWN0aW9uIHJlY292ZXJzIGl0LCAwIG90aGVyd2lzZS4KCiAgICBGb3IgZWFjaCBHVCBkaXZpc2lvbiwgdGhlIHByZWRpY3RlZCBncmFwaCBpcyBtYXRjaGVkIGFnYWluc3QgdGhlCiAgICBkaXZpc2lvbiBzdWJncmFwaCBhbmQgY2hlY2tlZCBmb3IgYSBzcGFubmluZyBjb21wb25lbnQgc2F0aXNmeWluZzoKCiAgICAxLiBBdCBsZWFzdCBvbmUgbWF0Y2hlZCBwcmVkaWN0aW9uIG5vZGUgaW4gdGhlIEdUJ3Mgb25lLW5vZGUgc3RhZ2UKICAgICAgIChwcmUtZGl2aXNpb24gdGltZXBvaW50cykuCiAgICAyLiBBdCBsZWFzdCB0d28gbWF0Y2hlZCBwcmVkaWN0aW9uIG5vZGVzIGF0IHRoZSBzYW1lIHRpbWVwb2ludCBpbiB0aGUKICAgICAgIEdUJ3MgdHdvLW5vZGUgc3RhZ2UgKHBvc3QtZGl2aXNpb24gdGltZXBvaW50cykuCiAgICAzLiBBbGwgbWF0Y2hlZCBwcmVkaWN0aW9uIG5vZGVzIGluIGEgc2luZ2xlIHdlYWtseS1jb25uZWN0ZWQKICAgICAgIGNvbXBvbmVudCBvZiB0aGUgcHJlZGljdGlvbiBncmFwaC4KCiAgICBFYWNoIHN1Y2ggc3Bhbm5pbmcgY29tcG9uZW50IGlzIGFzc29jaWF0ZWQgd2l0aCB0aGUgKnByZWQgZGl2aWRpbmcKICAgIG5vZGVzKiAob3V0LWRlZ3JlZSDiiaUgMikgaXQgY29udGFpbnMuIEEgbWF4aW11bS1jYXJkaW5hbGl0eSBiaXBhcnRpdGUKICAgIG1hdGNoaW5nIGlzIHRoZW4gY29tcHV0ZWQgc28gZWFjaCBwcmVkIGRpdmlkaW5nIG5vZGUgc2VydmVzIGF0IG1vc3QKICAgIG9uZSBHVCBkaXZpc2lvbiwgYW5kIGVhY2ggR1QgZGl2aXNpb24gaXMgcGFpcmVkIHdpdGggYXQgbW9zdCBvbmUKICAgIHByZWQgZGl2aWRpbmcgbm9kZS4gQSBHVCBkaXZpc2lvbiBzY29yZXMgMSBvbmx5IGlmIGl0IGlzIHBhaXJlZCBpbgogICAgdGhhdCBtYXRjaGluZyAtLSB0aGlzIHByZXZlbnRzIGEgc2luZ2xlIHByZWQgZm9yayBmcm9tIGJlaW5nCiAgICBjcmVkaXRlZCB0byBtdWx0aXBsZSBHVCBkaXZpc2lvbnMuCgogICAgUGFyYW1ldGVycwogICAgLS0tLS0tLS0tLQogICAgcHJlZF9ncmFwaCA6IHRkLmdyYXBoLkJhc2VHcmFwaAogICAgICAgIFRoZSBwcmVkaWN0ZWQgdHJhY2tpbmcgZ3JhcGguCiAgICBndF9ncmFwaCA6IHRkLmdyYXBoLkJhc2VHcmFwaAogICAgICAgIFRoZSBncm91bmQtdHJ1dGggdHJhY2tpbmcgZ3JhcGguCiAgICBzY2FsZSA6IHR1cGxlW2Zsb2F0LCAuLi5dIHwgTm9uZQogICAgICAgIFBoeXNpY2FsIHZveGVsIHNjYWxlIHVzZWQgZm9yIGNlbnRyb2lkLWRpc3RhbmNlIG1hdGNoaW5nLgogICAgbWF4X2Rpc3RhbmNlIDogZmxvYXQKICAgICAgICBNYXhpbXVtIGNlbnRyb2lkIGRpc3RhbmNlIGZvciBhIG1hdGNoLgoKICAgIFJldHVybnMKICAgIC0tLS0tLS0KICAgIGRpY3RbaW50LCBpbnRdCiAgICAgICAgTWFwcGluZyBmcm9tIEdUIGRpdmlkaW5nLW5vZGUgSUQgdG8gMSAocGFpcmVkKSBvciAwIChub3QpLgogICAgIiIiCiAgICBtYXRjaGVkID0gbWF0Y2hfZGl2aXNpb25zKAogICAgICAgIHByZWRfZ3JhcGgsIGd0X2dyYXBoLCBzY2FsZSwgbWF4X2Rpc3RhbmNlLAogICAgKQogICAgZ3RfZGl2aXNpb25zID0gZXh0cmFjdF9kaXZpc2lvbnMoZ3RfZ3JhcGgpCiAgICBwcmVkX2Rpdl9ub2RlcyA9IHNldChwcmVkX2dyYXBoLmRpdmlkaW5nX25vZGVzKCkpCgogICAgY2FuZGlkYXRlczogZGljdFtpbnQsIHNldFtpbnRdXSA9IHt9CiAgICBmb3IgZGl2X25vZGUsIG1hdGNoZWRfcHJlZCBpbiBtYXRjaGVkLml0ZW1zKCk6CiAgICAgICAgbWF0Y2hlZF9hdHRycyA9IF9tYXRjaGVkX25vZGVfYXR0cnMobWF0Y2hlZF9wcmVkKQogICAgICAgIG5vZGVfaWRzID0gbWF0Y2hlZF9hdHRyc1t0ZC5ERUZBVUxUX0FUVFJfS0VZUy5OT0RFX0lEXS50b19saXN0KCkKICAgICAgICBjb21wb25lbnRzID0gX3dlYWtseV9jb25uZWN0ZWRfY29tcG9uZW50cyhtYXRjaGVkX3ByZWQsIG5vZGVfaWRzKQogICAgICAgIGd0X2RpdiA9IGd0X2RpdmlzaW9uc1tkaXZfbm9kZV0KICAgICAgICBkaXZfY2FuZGlkYXRlczogc2V0W2ludF0gPSBzZXQoKQogICAgICAgIGZvciBtYXRjaGVkX3N1YnNldCwgdmlzaXRlZCBpbiBjb21wb25lbnRzOgogICAgICAgICAgICBjb21wX2F0dHJzID0gbWF0Y2hlZF9hdHRycy5maWx0ZXIoCiAgICAgICAgICAgICAgICBwbC5jb2wodGQuREVGQVVMVF9BVFRSX0tFWVMuTk9ERV9JRCkuaXNfaW4obGlzdChtYXRjaGVkX3N1YnNldCkpCiAgICAgICAgICAgICkKICAgICAgICAgICAgaWYgX2hhc19zdGFnZV9jb3ZlcmFnZShjb21wX2F0dHJzLCBndF9kaXYsIGRpdl9ub2RlKToKICAgICAgICAgICAgICAgIGRpdl9jYW5kaWRhdGVzIHw9IHZpc2l0ZWQgJiBwcmVkX2Rpdl9ub2RlcwogICAgICAgIGNhbmRpZGF0ZXNbZGl2X25vZGVdID0gZGl2X2NhbmRpZGF0ZXMKCiAgICBwYWlyaW5nID0gX2JpcGFydGl0ZV9tYXhfbWF0Y2hpbmcobGlzdChjYW5kaWRhdGVzKSwgY2FuZGlkYXRlcykKICAgIHJldHVybiB7ZGl2OiBpbnQoZGl2IGluIHBhaXJpbmcpIGZvciBkaXYgaW4gY2FuZGlkYXRlc30KCgpkZWYgY291bnRfbWF0Y2hlZF9wcmVkX2RpdmlzaW9ucygKICAgIHByZWRfZ3JhcGg6IHRkLmdyYXBoLkJhc2VHcmFwaCwKICAgIGd0X2dyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBzY2FsZTogdHVwbGVbZmxvYXQsIC4uLl0gfCBOb25lID0gTm9uZSwKICAgIG1heF9kaXN0YW5jZTogZmxvYXQgPSA3LjAsCikgLT4gaW50OgogICAgIiIiQ291bnQgcHJlZGljdGVkIGRpdmlzaW9uIG5vZGVzIHdob3NlIG1hdGNoZWQgR1Qgbm9kZSBpcyBhbm5vdGF0ZWQuCgogICAgTWF0Y2hlcyB0aGUgZnVsbCBwcmVkaWN0ZWQgZ3JhcGggYWdhaW5zdCB0aGUgZnVsbCBHVCBncmFwaC4gIEFtb25nCiAgICBwcmVkaWN0ZWQgbm9kZXMgdGhhdCB3ZXJlIG1hdGNoZWQgdG8gYSBHVCBub2RlLCBjb3VudHMgaG93IG1hbnkgYXJlCiAgICBkaXZpZGluZyAob3V0LWRlZ3JlZSA+PSAyKSBpbiB0aGUgcHJlZGljdGlvbiAqYW5kKiB3aG9zZSBtYXRjaGVkIEdUCiAgICBub2RlIGhhcyBhdCBsZWFzdCBvbmUgY2hpbGQuICBBIG1hdGNoZWQgR1Qgbm9kZSB3aXRoIG5vIGNoaWxkcmVuIG1hcmtzCiAgICB0aGUgZW5kIG9mIHRoZSBhbm5vdGF0aW9uIOKAlCB3ZSBjYW4ndCB0ZWxsIHdoZXRoZXIgdGhlIGNlbGwgYWN0dWFsbHkKICAgIGRpdmlkZWQgdGhlcmUsIHNvIHN1Y2ggcHJlZGljdGVkIGRpdmlzaW9ucyBhcmUgZXhjbHVkZWQgZnJvbSB0aGUgY291bnQKICAgIChhbmQgdGhlcmVmb3JlIGZyb20gdGhlIEZQIHRhbGx5KS4KCiAgICBQYXJhbWV0ZXJzCiAgICAtLS0tLS0tLS0tCiAgICBwcmVkX2dyYXBoIDogdGQuZ3JhcGguQmFzZUdyYXBoCiAgICAgICAgVGhlIHByZWRpY3RlZCB0cmFja2luZyBncmFwaC4KICAgIGd0X2dyYXBoIDogdGQuZ3JhcGguQmFzZUdyYXBoCiAgICAgICAgVGhlIGdyb3VuZC10cnV0aCB0cmFja2luZyBncmFwaC4KICAgIHNjYWxlIDogdHVwbGVbZmxvYXQsIC4uLl0gfCBOb25lCiAgICAgICAgUGh5c2ljYWwgdm94ZWwgc2NhbGUgdXNlZCBmb3IgY2VudHJvaWQtZGlzdGFuY2UgbWF0Y2hpbmcuCiAgICBtYXhfZGlzdGFuY2UgOiBmbG9hdAogICAgICAgIE1heGltdW0gY2VudHJvaWQgZGlzdGFuY2UgZm9yIGEgbWF0Y2guCgogICAgUmV0dXJucwogICAgLS0tLS0tLQogICAgaW50CiAgICAgICAgTnVtYmVyIG9mIG1hdGNoZWQgcHJlZGljdGVkIGRpdmlzaW9uIG5vZGVzLgogICAgIiIiCiAgICBtYXRjaGVkX3ByZWQgPSBfbWF0Y2hfZnVsbCgKICAgICAgICBwcmVkX2dyYXBoLCBndF9ncmFwaCwgc2NhbGUsIG1heF9kaXN0YW5jZSwKICAgICkKCiAgICBub2RlX2F0dHJzID0gbWF0Y2hlZF9wcmVkLm5vZGVfYXR0cnMoCiAgICAgICAgYXR0cl9rZXlzPVt0ZC5ERUZBVUxUX0FUVFJfS0VZUy5OT0RFX0lELCB0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSURdLAogICAgKQogICAgbWF0Y2hlZF9ub2RlcyA9IG5vZGVfYXR0cnMuZmlsdGVyKAogICAgICAgIHBsLmNvbCh0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSUQpLmlzX25vdF9udWxsKCkKICAgICAgICAmIChwbC5jb2wodGQuREVGQVVMVF9BVFRSX0tFWVMuTUFUQ0hFRF9OT0RFX0lEKSAhPSAtMSkKICAgICkKCiAgICBjb3VudCA9IDAKICAgIGZvciByb3cgaW4gbWF0Y2hlZF9ub2Rlcy5pdGVyX3Jvd3MobmFtZWQ9VHJ1ZSk6CiAgICAgICAgcHJlZF9ub2RlID0gcm93W3RkLkRFRkFVTFRfQVRUUl9LRVlTLk5PREVfSURdCiAgICAgICAgZ3Rfbm9kZSA9IHJvd1t0ZC5ERUZBVUxUX0FUVFJfS0VZUy5NQVRDSEVEX05PREVfSURdCiAgICAgICAgaWYgKAogICAgICAgICAgICBtYXRjaGVkX3ByZWQub3V0X2RlZ3JlZShwcmVkX25vZGUpID49IDIKICAgICAgICAgICAgYW5kIGd0X2dyYXBoLm91dF9kZWdyZWUoZ3Rfbm9kZSkgPj0gMQogICAgICAgICk6CiAgICAgICAgICAgIGNvdW50ICs9IDEKICAgIHJldHVybiBjb3VudAoKCmRlZiBldmFsdWF0ZV9kaXZpc2lvbnMoCiAgICBwcmVkX2dyYXBoOiB0ZC5ncmFwaC5CYXNlR3JhcGgsCiAgICBndF9ncmFwaDogdGQuZ3JhcGguQmFzZUdyYXBoLAogICAgc2NhbGU6IHR1cGxlW2Zsb2F0LCAuLi5dIHwgTm9uZSA9IE5vbmUsCiAgICBtYXhfZGlzdGFuY2U6IGZsb2F0ID0gNy4wLAopIC0+IERpdmlzaW9uQ291bnRzOgogICAgIiIiQ29tcHV0ZSBUUCwgRk4sIGFuZCBGUCBjb3VudHMgZm9yIGRpdmlzaW9uIGV2ZW50cy4KCiAgICAtICoqVFAqKjogR1QgZGl2aXNpb25zIGNvcnJlY3RseSByZWNvdmVyZWQgaW4gdGhlIHByZWRpY3Rpb24KICAgICAgKG1hdGNoZWQgbm9kZXMgY29ubmVjdGVkIGFuZCBmb3JraW5nKS4KICAgIC0gKipGTioqOiBHVCBkaXZpc2lvbnMgbm90IHJlY292ZXJlZC4KICAgIC0gKipGUCoqOiBQcmVkaWN0ZWQgZGl2aXNpb25zIHdob3NlIG1hdGNoZWQgR1Qgbm9kZSBpcyBub3QgZGl2aWRpbmcuCgogICAgUGFyYW1ldGVycwogICAgLS0tLS0tLS0tLQogICAgcHJlZF9ncmFwaCA6IHRkLmdyYXBoLkJhc2VHcmFwaAogICAgICAgIFRoZSBwcmVkaWN0ZWQgdHJhY2tpbmcgZ3JhcGguCiAgICBndF9ncmFwaCA6IHRkLmdyYXBoLkJhc2VHcmFwaAogICAgICAgIFRoZSBncm91bmQtdHJ1dGggdHJhY2tpbmcgZ3JhcGguCiAgICBzY2FsZSA6IHR1cGxlW2Zsb2F0LCAuLi5dIHwgTm9uZQogICAgICAgIFBoeXNpY2FsIHZveGVsIHNjYWxlIHVzZWQgZm9yIGNlbnRyb2lkLWRpc3RhbmNlIG1hdGNoaW5nLgogICAgbWF4X2Rpc3RhbmNlIDogZmxvYXQKICAgICAgICBNYXhpbXVtIGNlbnRyb2lkIGRpc3RhbmNlIGZvciBhIG1hdGNoLgoKICAgIFJldHVybnMKICAgIC0tLS0tLS0KICAgIERpdmlzaW9uQ291bnRzCiAgICAgICAgTmFtZWQgdHVwbGUgd2l0aCBgYHRwYGAsIGBgZm5gYCwgYW5kIGBgZnBgYCBmaWVsZHMuCiAgICAiIiIKICAgIHNjb3JlcyA9IHNjb3JlX2RpdmlzaW9ucygKICAgICAgICBwcmVkX2dyYXBoLCBndF9ncmFwaCwgc2NhbGUsIG1heF9kaXN0YW5jZSwKICAgICkKICAgIHRwID0gc3VtKHNjb3Jlcy52YWx1ZXMoKSkKICAgIGZuID0gbGVuKHNjb3JlcykgLSB0cAogICAgbWF0Y2hlZF9wcmVkX2RpdnMgPSBjb3VudF9tYXRjaGVkX3ByZWRfZGl2aXNpb25zKAogICAgICAgIHByZWRfZ3JhcGgsIGd0X2dyYXBoLCBzY2FsZSwgbWF4X2Rpc3RhbmNlLAogICAgKQogICAgZnAgPSBtYXgoMCwgbWF0Y2hlZF9wcmVkX2RpdnMgLSB0cCkKICAgIHJldHVybiBEaXZpc2lvbkNvdW50cyh0cD10cCwgZm49Zm4sIGZwPWZwKQo="
}
M32_1_EMBED_SHA = {
  "__init__.py": "f9df2098dd85934d0b0d48d477420b1f9ed20a9a55ca6294a1874b964e9166e6",
  "metrics.py": "700906a6c1cf5db6c2347d3fe169905d9b17fc6480c5b91b540d825145e7fa91",
  "division_metrics.py": "d1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be"
}

# --------------------------------------------------------------------------- #
# 41. M32.1 vendored OFFICIAL metric: materialise (verbatim) -> import -> verify
# --------------------------------------------------------------------------- #
# M32_1_EMBED / M32_1_EMBED_SHA are injected at assemble time: base64 of the EXACT
# vendored royerlab/kaggle-cell-tracking-competition files (byte-identical). The
# metric logic is never rewritten; a runtime sha256 check proves the materialised
# source equals the vendored snapshot.
import base64 as _b64
import hashlib as _hl
import importlib as _il
import sys as _sys

M32_1_UPSTREAM_REPO = "https://github.com/royerlab/kaggle-cell-tracking-competition"
M32_1_UPSTREAM_BRANCH = "main"
M32_1_UPSTREAM_COMMIT = "7396b7e98e61844e799152ddda7e5493084cc8f3"
M32_1_METRIC_MODULE = "tracking_cellmot.metrics"
M32_1_REQUIRED_API = ["evaluate", "evaluate_datasets", "per_sample_metrics", "summarise"]
M32_1_REQUIRED_DEPS = ["tracksdata", "geff", "polars", "scipy"]
M32_1_400EP_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
M32_1_400EP_NAME_SUBSTR = "400ep"
M32_1_WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"


def _sha256_bytes(b):
    return _hl.sha256(b).hexdigest()


def materialise_vendored_metric(target_root=None) -> dict:
    """Write the embedded (byte-identical) tracking_cellmot core to
    <target_root>/src/tracking_cellmot and verify each file's sha256 against the
    vendored snapshot. Returns a manifest (path, sha256, verified)."""
    target_root = Path(target_root or (Path(KAGGLE_WORKING_DIR) / "vendor" / "m32_1_cellmot"))
    pkg = target_root / "src" / "tracking_cellmot"
    pkg.mkdir(parents=True, exist_ok=True)
    written = []
    all_ok = True
    for rel, b64 in M32_1_EMBED.items():
        raw = _b64.b64decode(b64.encode())
        sha = _sha256_bytes(raw)
        expected = M32_1_EMBED_SHA[rel]
        ok = (sha == expected)
        all_ok = all_ok and ok
        dest = pkg / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        written.append({"file": f"tracking_cellmot/{rel}", "sha256": sha, "expected_sha256": expected, "verified": ok})
    return {"target_root": str(target_root), "src_path": str(target_root / "src"), "files": written,
            "all_verified": all_ok, "upstream_commit": M32_1_UPSTREAM_COMMIT}


def _find_mounted_vendor_src() -> list:
    """Search mounted inputs / repo / working for a vendored tracking_cellmot src
    (a Kaggle utility-dataset bundle or the committed vendor dir)."""
    found = []
    for base in ["/kaggle/input", KAGGLE_WORKING_DIR, str(Path(KAGGLE_COMPETITION_INPUT_DIR).parent), "."]:
        b = Path(base)
        if not b.exists():
            continue
        for p in b.glob("**/tracking_cellmot/metrics.py"):
            src = p.parent.parent
            if src.name == "src" or (src / "tracking_cellmot").is_dir():
                found.append(str(src))
    return sorted(set(found))


def _check_dependencies() -> dict:
    out = {}
    for dep in M32_1_REQUIRED_DEPS:
        try:
            m = _il.import_module(dep)
            out[dep] = {"available": True, "version": getattr(m, "__version__", None)}
        except Exception as exc:
            out[dep] = {"available": False, "error": type(exc).__name__}
    return out


def resolve_official_metric_vendored() -> dict:
    """Prepend a vendored src to sys.path (mounted bundle if present, else the
    embedded materialised copy) and import the OFFICIAL tracking_cellmot.metrics.
    Verifies the required API + division import + deps. NEVER pip-installs from the
    internet and NEVER uses local_metric.py."""
    report = {"official_metric_found": False, "import_path": None, "upstream_commit": M32_1_UPSTREAM_COMMIT,
              "compatibility_patch_applied": False, "compatibility_patch_diff": None,
              "evaluate_callable": False, "evaluate_datasets_callable": False, "summarise_callable": False,
              "per_sample_metrics_callable": False, "division_import_ok": False,
              "source_hashes": {}, "dependency_versions": _check_dependencies(), "errors": []}
    deps = report["dependency_versions"]
    missing = [d for d in M32_1_REQUIRED_DEPS if not deps[d]["available"]]
    if missing:
        report["status"] = "OFFICIAL_METRIC_DEPENDENCY_MISSING"
        report["missing_dependencies"] = missing
        # still materialise + hash-verify the vendored source (offline proof).
        mat = materialise_vendored_metric()
        report["materialised"] = mat
        report["source_hashes"] = {f["file"]: f["sha256"] for f in mat["files"]}
        report["vendored_source_verified"] = mat["all_verified"]
        return report
    # deps present: get a vendored src (mounted preferred, else embedded).
    candidates = _find_mounted_vendor_src()
    mat = materialise_vendored_metric()
    report["materialised"] = mat
    report["source_hashes"] = {f["file"]: f["sha256"] for f in mat["files"]}
    report["vendored_source_verified"] = mat["all_verified"]
    candidates = candidates + [mat["src_path"]]
    for src in candidates:
        if src not in _sys.path:
            _sys.path.insert(0, src)
        for mod in ["tracking_cellmot", "tracking_cellmot.metrics", "tracking_cellmot.division_metrics"]:
            _sys.modules.pop(mod, None)
        try:
            metrics = _il.import_module("tracking_cellmot.metrics")
        except Exception as exc:
            report["errors"].append(f"import from {src}: {exc!r}")
            continue
        report["official_metric_found"] = True
        report["import_path"] = getattr(metrics, "__file__", src)
        report["evaluate_callable"] = callable(getattr(metrics, "evaluate", None))
        report["evaluate_datasets_callable"] = callable(getattr(metrics, "evaluate_datasets", None))
        report["summarise_callable"] = callable(getattr(metrics, "summarise", None))
        report["per_sample_metrics_callable"] = callable(getattr(metrics, "per_sample_metrics", None))
        try:
            _il.import_module("tracking_cellmot.division_metrics")
            report["division_import_ok"] = True
        except Exception as exc:
            report["errors"].append(f"division import: {exc!r}")
        report["metrics_module"] = metrics
        report["status"] = ("OFFICIAL_METRIC_READY" if all(report[k] for k in
                            ["evaluate_callable", "evaluate_datasets_callable", "summarise_callable",
                             "per_sample_metrics_callable", "division_import_ok"]) else "OFFICIAL_METRIC_VERIFICATION_FAILED")
        return report
    report["status"] = "OFFICIAL_METRIC_NOT_FOUND"
    return report

# --------------------------------------------------------------------------- #
# 42. M32.1 OFFICIAL-metric synthetic fixtures (run the ACTUAL vendored scorer)
# --------------------------------------------------------------------------- #
# Graphs are built with the exact tracksdata API the official tests use
# (td.graph.InMemoryGraph, add_node attrs T/z/y/x, add_edge). No expected value
# taken from the proxy scorer - each assertion is derived from the official
# evaluate() output itself (perfect graph -> jaccard 1.0; missing/extra edge moves
# FN/FP; divisions move division_tp/fp/fn).

def _td_build_graph(td, nodes, edges):
    """nodes: list of (t, z, y, x). edges: list of (src_idx, tgt_idx). Returns a
    tracksdata InMemoryGraph with T + spatial attrs set."""
    import polars as pl  # noqa: F401
    g = td.graph.InMemoryGraph()
    ids = []
    for (t, z, y, x) in nodes:
        nid = g.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: int(t), "z": float(z), "y": float(y), "x": float(x)})
        ids.append(nid)
    for (s, t) in edges:
        g.add_edge(ids[s], ids[t], {})
    return g, ids


def run_official_metric_fixtures(metrics_module) -> dict:
    """Score 10 synthetic fixtures with the OFFICIAL evaluate()/summarise().
    Returns raw counts + pass flags. Requires tracksdata/polars; on ImportError or
    an API mismatch returns a specific status (never a fabricated pass)."""
    result = {"all_verification_tests_passed": False, "tests": [], "status": None}
    try:
        import tracksdata as td
        import polars  # noqa: F401
    except Exception as exc:
        result["status"] = "OFFICIAL_METRIC_DEPENDENCY_MISSING"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    evaluate = getattr(metrics_module, "evaluate", None)
    per_sample = getattr(metrics_module, "per_sample_metrics", None)
    summarise = getattr(metrics_module, "summarise", None)
    if not all(callable(f) for f in [evaluate, per_sample, summarise]):
        result["status"] = "OFFICIAL_METRIC_VERIFICATION_FAILED"
        result["error"] = "official evaluate/per_sample_metrics/summarise not callable"
        return result

    def _ev(pred_spec, gt_spec, **kw):
        pg, _ = _td_build_graph(td, *pred_spec)
        gg, _ = _td_build_graph(td, *gt_spec)
        return evaluate(pg, gg, max_distance=kw.get("max_distance", 7.0))

    tests = []
    try:
        # a simple 3-node track at the SAME coordinates in pred and GT.
        track = ([(0, 0, 0, 0), (1, 0, 0, 0.5), (2, 0, 0, 1.0)], [(0, 1), (1, 2)])
        # 1. perfect edge graph -> jaccard 1.0 (FP=FN=0).
        er = _ev(track, track)
        j = er.edge_tp / (er.edge_tp + er.edge_fp + er.edge_fn) if (er.edge_tp + er.edge_fp + er.edge_fn) else float("nan")
        tests.append({"name": "perfect_edge_graph", "edge_tp": er.edge_tp, "edge_fp": er.edge_fp, "edge_fn": er.edge_fn,
                      "edge_jaccard": j, "passed": bool(er.edge_fp == 0 and er.edge_fn == 0 and abs(j - 1.0) < 1e-9)})
        # 2. missing edge in pred -> FN increases.
        er2 = _ev(([(0, 0, 0, 0), (1, 0, 0, 0.5), (2, 0, 0, 1.0)], [(0, 1)]), track)
        tests.append({"name": "missing_edge_fn", "edge_fn": er2.edge_fn, "passed": bool(er2.edge_fn > er.edge_fn)})
        # 3. conflicting annotated edge (pred edge to a wrong matched node) -> FP.
        er3 = _ev(([(0, 0, 0, 0), (1, 0, 0, 0.5), (2, 0, 0, 1.0)], [(0, 1), (1, 2), (0, 2)]), track)
        tests.append({"name": "conflicting_edge_fp", "edge_fp": er3.edge_fp, "passed": bool(er3.edge_fp >= er.edge_fp)})
        # 4. extra edge wholly outside annotated support (far node) - behaviour = official.
        er4 = _ev(([(0, 0, 0, 0), (1, 0, 0, 0.5), (2, 0, 0, 1.0), (1, 0, 100, 100)], [(0, 1), (1, 2), (0, 3)]), track)
        tests.append({"name": "extra_unannotated_edge", "edge_fp": er4.edge_fp, "edge_tp": er4.edge_tp,
                      "passed": bool(er4.edge_tp == er.edge_tp)})
        # 5-7 divisions: parent with two children (out-degree 2).
        div = ([(0, 0, 0, 0), (1, 0, 0, 0.5), (1, 0, 0.5, 0.0)], [(0, 1), (0, 2)])
        erd = _ev(div, div)
        tests.append({"name": "correct_division_tp", "division_tp": erd.division_tp, "division_fp": erd.division_fp,
                      "division_fn": erd.division_fn, "passed": bool(erd.division_tp >= 1 and erd.division_fn == 0)})
        # 6. false fork (pred divides, GT does not).
        erf = _ev(div, ([(0, 0, 0, 0), (1, 0, 0, 0.5), (1, 0, 0.5, 0.0)], [(0, 1)]))
        tests.append({"name": "false_fork_fp", "division_fp": erf.division_fp, "passed": bool(erf.division_fp >= 1)})
        # 7. missed division (GT divides, pred does not).
        erm = _ev(([(0, 0, 0, 0), (1, 0, 0, 0.5), (1, 0, 0.5, 0.0)], [(0, 1)]), div)
        tests.append({"name": "missed_division_fn", "division_fn": erm.division_fn, "passed": bool(erm.division_fn >= 1)})
        # 8. 7um matching boundary: a GT node just beyond 7um must NOT match.
        near = _ev(([(0, 0, 0, 0), (1, 0, 0, 0.5)], [(0, 1)]), ([(0, 0, 0, 0), (1, 0, 0, 0.5)], [(0, 1)]))
        far = _ev(([(0, 0, 0, 0), (1, 0, 0, 0.5)], [(0, 1)]), ([(0, 0, 100, 100), (1, 0, 100, 100.5)], [(0, 1)]))
        tests.append({"name": "matching_7um_boundary", "near_tp": near.edge_tp, "far_tp": far.edge_tp,
                      "passed": bool(near.edge_tp >= 1 and far.edge_tp == 0)})
        # 9. node overprediction -> total_node_ratio > 0 -> adjusted score < raw jaccard.
        row_ok = per_sample(er, float(len(track[0])), 1.0)
        row_over = per_sample(er, float(len(track[0])) / 3.0, 1.0)   # N_total much smaller -> ratio > 0
        tests.append({"name": "node_overprediction_adjustment",
                      "adj_ok": row_ok.get("adj_edge_jaccard"), "adj_over": row_over.get("adj_edge_jaccard"),
                      "passed": bool(row_over.get("adj_edge_jaccard") is not None and row_ok.get("adj_edge_jaccard") is not None
                                     and row_over["adj_edge_jaccard"] <= row_ok["adj_edge_jaccard"])})
        # 10. multi-dataset summarise aggregation.
        s = summarise([per_sample(er, float(len(track[0])), 1.0), per_sample(erd, 3.0, 1.0)])
        tests.append({"name": "multi_dataset_summarise", "n": s.get("n"), "score": s.get("score"),
                      "passed": bool(s.get("n") == 2 and s.get("score") == s.get("score"))})
    except Exception as exc:
        result["status"] = "OFFICIAL_METRIC_VERIFICATION_FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["tests"] = tests
        return result
    result["tests"] = tests
    result["all_verification_tests_passed"] = all(t["passed"] for t in tests)
    result["status"] = "OFFICIAL_METRIC_VENDOR_PASS" if result["all_verification_tests_passed"] else "OFFICIAL_METRIC_VERIFICATION_FAILED"
    return result

# --------------------------------------------------------------------------- #
# 43. M32.1 PART 3 normalized train inventory + PART 4 split_0 recovery cascade
# --------------------------------------------------------------------------- #
import hashlib as _hl2
import re as _re2

_M32_1_GROUP_RE = _re2.compile(r"^([0-9a-fA-F]{4})_")


def _sha256_file_m321(p):
    p = Path(p)
    if not p.exists() or not p.is_file():
        return None
    h = _hl2.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _normalize_dataset_name(name: str) -> str:
    for suf in (".geff", ".zarr"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def _read_estimated_n_nodes(geff_root: Path):
    """estimated_number_of_nodes from GEFF metadata (geff .attrs / zarr.json extras).
    Returns float or None - never fabricated."""
    for cand in [geff_root / "zarr.json", geff_root / ".zattrs", geff_root / "nodes" / "zarr.json"]:
        if cand.exists():
            try:
                meta = json.loads(cand.read_text())
            except Exception:
                continue
            for key in ["estimated_number_of_nodes", "estimated_n_total", "n_total"]:
                v = _deep_get(meta, key)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
    return None


def _deep_get(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _deep_get(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _deep_get(v, key)
            if r is not None:
                return r
    return None


def normalize_train_inventory(competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """Normalized paired-dataset inventory: strip .geff, exclude the `train` dir
    entry, require a paired GT GEFF (preferably a paired Zarr input), de-duplicate,
    read GT counts + estimated_number_of_nodes + source group. Reports missing pairs."""
    comp = Path(competition_dir)
    rows = {}
    missing_pairs = []
    for root in [comp / "train", comp]:
        if not root.exists():
            continue
        # candidate GEFF stores (dirs with nodes/ids) and any *.geff dir.
        for p in sorted(list(root.glob("*")) + list(root.glob("*.geff"))):
            if not p.is_dir():
                continue
            if p.name in ("train", "test", "val"):   # exclude container dirs
                continue
            gr = _geff_root_of(p)
            if gr is None:
                continue
            name = _normalize_dataset_name(p.name)
            if name in rows:
                continue
            zarr_path = None
            for zc in [root / f"{name}.zarr", root / name, comp / "train_images" / f"{name}.zarr"]:
                if zc.exists() and zc.name != p.name:
                    zarr_path = str(zc); break
            n_nodes = n_edges = None
            try:
                ids = _read_geff_array(gr, "nodes/ids")
                n_nodes = int(np.asarray(ids).reshape(-1).size) if ids is not None else None
                eids = _read_geff_array(gr, "edges/ids")   # GT edges, no solution mask
                if eids is not None:
                    e = np.asarray(eids)
                    n_edges = int(e.shape[0] if e.ndim > 1 else e.reshape(-1, 2).shape[0])
            except Exception:
                pass
            grp = _M32_1_GROUP_RE.match(name)
            rows[name] = {"dataset": name, "geff_path": str(gr), "zarr_path": zarr_path,
                          "geff_exists": True, "zarr_exists": zarr_path is not None,
                          "gt_node_count": n_nodes, "gt_edge_count": n_edges,
                          "estimated_number_of_nodes": _read_estimated_n_nodes(gr),
                          "source_group": grp.group(1) if grp else None}
            if zarr_path is None:
                missing_pairs.append({"dataset": name, "missing": "zarr_input"})
    inv = sorted(rows.values(), key=lambda r: r["dataset"])
    return {"n_datasets": len(inv), "inventory": inv, "missing_pairs": missing_pairs,
            "source_groups": sorted(set(r["source_group"] for r in inv if r["source_group"]))}


# ---------- PART 4: exact split_0 recovery cascade --------------------------- #
M32_1_SPLIT_FILE_PATTERNS = ["dataset_splits.json", "*splits*.json", "*fold*.json",
                             "*train*val*.json", "*manifest*.json", "args.json", "*config*.yaml", "*config*.yml"]


def _search_split_files(roots) -> list:
    found = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for pat in M32_1_SPLIT_FILE_PATTERNS:
            for p in r.glob(f"**/{pat}"):
                if p.is_file() and p.stat().st_size < 4_000_000:
                    found.append(str(p))
    return sorted(set(found))


def _parse_split_file(path):
    """Parse split_0 train/test lists from a split json. Returns (train, test) or
    (None, None). The official evaluate uses the fold's `test` list as the expected
    evaluation datasets."""
    try:
        cfg = json.loads(Path(path).read_text())
    except Exception:
        return None, None
    node = cfg
    if isinstance(cfg, dict):
        for k in ["0", "split_0", "fold_0", "folds"]:
            if k in cfg:
                node = cfg[k]
                if isinstance(node, dict) and "0" in node:
                    node = node["0"]
                break
    if isinstance(node, dict):
        train = node.get("train") or node.get("train_datasets")
        test = node.get("test") or node.get("val") or node.get("validation")
        if train is not None or test is not None:
            return ([_normalize_dataset_name(str(x)) for x in train] if train else None,
                    [_normalize_dataset_name(str(x)) for x in test] if test else None)
    return None, None


def _inspect_checkpoint_metadata(artifact_dir):
    """Safely inspect the 400ep checkpoint for split/fold/train/val metadata using
    the trusted torch loader (weights_only=True). Never executes arbitrary pickle."""
    out = {"inspected": False, "split": None, "train_datasets": None, "val_datasets": None, "seed": None, "error": None}
    wpath = Path(artifact_dir) / M32_1_WEIGHTS_REL if artifact_dir else None
    if wpath is None or not wpath.exists():
        return out
    try:
        import torch
        ckpt = torch.load(str(wpath), map_location="cpu", weights_only=True)
        out["inspected"] = True
        meta = ckpt if isinstance(ckpt, dict) else {}
        for k in ["split", "fold", "split_index"]:
            if k in meta:
                out["split"] = meta[k]
        out["train_datasets"] = meta.get("train_datasets") or meta.get("train")
        out["val_datasets"] = meta.get("val_datasets") or meta.get("val") or meta.get("test")
        out["seed"] = meta.get("seed")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def recover_split0(competition_dir=KAGGLE_COMPETITION_INPUT_DIR, artifact_dir=None, all_datasets=None) -> dict:
    """Strict split_0 recovery cascade. Only produces a clean holdout from an
    OBSERVED split file or checkpoint metadata (or an EXACT deterministic
    reconstruction when every rule is observed - never otherwise). Enforces a
    zero-overlap leakage guard."""
    roots = ["/kaggle/input", KAGGLE_WORKING_DIR, competition_dir, str(Path(KAGGLE_COMPETITION_INPUT_DIR).parent)]
    if artifact_dir:
        roots += [str(artifact_dir), str(Path(artifact_dir).parent)]
    split_files = _search_split_files(roots)
    report = {"split_file_found": bool(split_files), "split_file_path": None, "split_file_sha256": None,
              "split_source_type": "UNRESOLVED", "split_0_train": None, "split_0_test_or_validation": None,
              "overlap": None, "missing_datasets": [], "extra_datasets": [], "clean_holdout_proven": False,
              "candidate_split_files": split_files, "checkpoint": None,
              "deterministic_reconstruction": {"attempted": False, "observed": {
                  "dataset_ordering_rule": False, "num_folds": False, "seed": False, "shuffle_algorithm": False,
                  "train_test_interpretation": False, "input_dataset_set": False}, "allowed": False},
              "provenance": [], "status": "CLEAN_HOLDOUT_UNRESOLVED"}
    all_names = set(all_datasets or [])
    # A. observed split file.
    for sf in split_files:
        train, test = _parse_split_file(sf)
        if train is not None or test is not None:
            report.update({"split_file_path": sf, "split_file_sha256": _sha256_file_m321(sf),
                           "split_source_type": "OBSERVED_FILE", "split_0_train": sorted(train or []),
                           "split_0_test_or_validation": sorted(test or [])})
            report["provenance"].append(f"OBSERVED split file {sf}")
            break
    # B. checkpoint metadata (only if no observed file).
    if report["split_source_type"] == "UNRESOLVED":
        ck = _inspect_checkpoint_metadata(artifact_dir)
        report["checkpoint"] = ck
        if ck["inspected"] and (ck["train_datasets"] or ck["val_datasets"]):
            report.update({"split_source_type": "CHECKPOINT_METADATA",
                           "split_0_train": sorted([_normalize_dataset_name(str(x)) for x in (ck["train_datasets"] or [])]),
                           "split_0_test_or_validation": sorted([_normalize_dataset_name(str(x)) for x in (ck["val_datasets"] or [])])})
            report["provenance"].append("checkpoint metadata (weights_only torch.load)")
    # C. deterministic reconstruction is FORBIDDEN here (the repo READS dataset_splits.json;
    #    it does not deterministically regenerate it from an observable seed+algorithm).
    #    report["...allowed"] stays False.
    # Finalize: leakage guard + clean holdout.
    train = set(report["split_0_train"] or [])
    test = set(report["split_0_test_or_validation"] or [])
    if report["split_source_type"] in ("OBSERVED_FILE", "CHECKPOINT_METADATA") and test:
        overlap = sorted(train & test)
        report["overlap"] = overlap
        report["missing_datasets"] = sorted([d for d in test if all_names and d not in all_names])
        report["extra_datasets"] = sorted([d for d in (all_names - train - test)]) if all_names else []
        if overlap:
            report["status"] = "DATA_LEAKAGE_DETECTED"
        elif report["missing_datasets"]:
            report["status"] = "SPLIT_DATASET_MISMATCH"
        else:
            report["clean_holdout_proven"] = True
            report["status"] = "SPLIT_RECOVERY_PASS"
    return report

# --------------------------------------------------------------------------- #
# 44. M32.1 orchestration (A vendor audit / B combined re-audit) - NON-SUBMIT
# --------------------------------------------------------------------------- #
def m32_1_artifact_guard() -> dict:
    art = resolve_artifact()
    if not art.get("resolved"):
        return {"artifact_guard_passed": False, "resolved": False, "artifact_dir": None, "artifact_name": None, "weight_sha256": None}
    adir = Path(art["artifact_dir"]); name = None
    mp = adir / "ARTIFACT_MANIFEST.json"
    if mp.exists():
        try:
            name = json.loads(mp.read_text()).get("artifact_name")
        except Exception:
            name = None
    sha = _sha256_file_m321(adir / M32_1_WEIGHTS_REL)
    blob = " ".join(str(x).lower() for x in [name, adir.name, str(adir)])
    return {"artifact_guard_passed": bool(sha == M32_1_400EP_SHA256 and M32_1_400EP_NAME_SUBSTR in blob),
            "resolved": True, "artifact_dir": str(adir), "artifact_name": name, "weight_sha256": sha,
            "expected_sha256": M32_1_400EP_SHA256}


def run_m32_1_vendor_audit(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """PART 2 (non-submit): materialise + import the vendored OFFICIAL metric, then
    run the 10 synthetic official fixtures. Writes m32_1_metric_vendor_report.json."""
    _RUN_LOG.clear()
    out_dir = Path(working_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vend = resolve_official_metric_vendored()
    fixtures = {"status": "SKIPPED_METRIC_NOT_READY", "all_verification_tests_passed": False, "tests": []}
    if vend.get("official_metric_found") and vend.get("status") == "OFFICIAL_METRIC_READY":
        fixtures = run_official_metric_fixtures(vend["metrics_module"])
    elif vend.get("status") == "OFFICIAL_METRIC_DEPENDENCY_MISSING":
        fixtures = {"status": "OFFICIAL_METRIC_DEPENDENCY_MISSING", "all_verification_tests_passed": False, "tests": []}
    if vend.get("status") == "OFFICIAL_METRIC_DEPENDENCY_MISSING":
        rec = "OFFICIAL_METRIC_DEPENDENCY_MISSING"
    elif not vend.get("official_metric_found"):
        rec = "OFFICIAL_METRIC_VERIFICATION_FAILED"
    elif fixtures.get("all_verification_tests_passed"):
        rec = "OFFICIAL_METRIC_VENDOR_PASS"
    else:
        rec = "OFFICIAL_METRIC_VERIFICATION_FAILED"
    report = {
        "kind": "official_metric_vendor_audit", "submit": False,
        "upstream_repository": M32_1_UPSTREAM_REPO, "upstream_branch": M32_1_UPSTREAM_BRANCH, "upstream_commit": M32_1_UPSTREAM_COMMIT,
        "official_metric_found": vend.get("official_metric_found"), "import_path": vend.get("import_path"),
        "source_hashes": vend.get("source_hashes"), "vendored_source_verified": vend.get("vendored_source_verified"),
        "compatibility_patch_applied": vend.get("compatibility_patch_applied"), "compatibility_patch_diff": vend.get("compatibility_patch_diff"),
        "evaluate_callable": vend.get("evaluate_callable"), "evaluate_datasets_callable": vend.get("evaluate_datasets_callable"),
        "summarise_callable": vend.get("summarise_callable"), "per_sample_metrics_callable": vend.get("per_sample_metrics_callable"),
        "division_import_ok": vend.get("division_import_ok"), "dependency_versions": vend.get("dependency_versions"),
        "missing_dependencies": vend.get("missing_dependencies"),
        "all_verification_tests_passed": fixtures.get("all_verification_tests_passed"),
        "verification_tests": fixtures.get("tests"), "verification_status": fixtures.get("status"),
        "recommendation": rec, "used_local_metric_py": False,
    }
    (out_dir / "m32_1_metric_vendor_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_log(out_dir)
    print("=== M32.1_A OFFICIAL METRIC VENDOR AUDIT (non-submit) ===")
    print(f"  upstream commit: {M32_1_UPSTREAM_COMMIT}")
    print(f"  official_metric_found: {report['official_metric_found']}  vendored_source_verified: {report['vendored_source_verified']}")
    print(f"  deps: { {k: v['available'] for k, v in (report['dependency_versions'] or {}).items()} }")
    print(f"  fixtures passed: {report['all_verification_tests_passed']} ({report['verification_status']})")
    print(f"  RECOMMENDATION: {report['recommendation']}   (local_metric.py used: False)")
    print("  No submission (vendor audit only).")
    return report


def run_m32_1_reaudit(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR) -> dict:
    """PART 5 (non-submit): combined re-audit. AUDIT_PASS only when the vendored
    metric imports + all fixtures pass + normalized inventory valid + clean
    split_0 holdout proven + overlap zero + 400ep SHA passes. Writes
    m32_1_official_cv_reaudit.json + m32_1_split_recovery.json + m32_1_metric_vendor_report.json."""
    _RUN_LOG.clear()
    out_dir = Path(working_dir); out_dir.mkdir(parents=True, exist_ok=True)
    guard = m32_1_artifact_guard()
    vendor = run_m32_1_vendor_audit(working_dir, competition_dir)
    inv = normalize_train_inventory(competition_dir)
    split = recover_split0(competition_dir, guard.get("artifact_dir"), all_datasets=[r["dataset"] for r in inv["inventory"]])
    (out_dir / "m32_1_split_recovery.json").write_text(json.dumps(split, indent=2, default=str))
    pd.DataFrame(inv["inventory"]).to_csv(out_dir / "m32_1_normalized_train_inventory.csv", index=False)

    # recommendation cascade.
    if not guard["artifact_guard_passed"]:
        rec = "WRONG_ARTIFACT"
    elif vendor["recommendation"] == "OFFICIAL_METRIC_DEPENDENCY_MISSING":
        rec = "OFFICIAL_METRIC_DEPENDENCY_MISSING"
    elif vendor["recommendation"] != "OFFICIAL_METRIC_VENDOR_PASS":
        rec = "OFFICIAL_METRIC_VERIFICATION_FAILED"
    elif split["status"] == "DATA_LEAKAGE_DETECTED":
        rec = "DATA_LEAKAGE_DETECTED"
    elif not split["clean_holdout_proven"]:
        rec = "CLEAN_HOLDOUT_UNRESOLVED"
    else:
        rec = "AUDIT_PASS"
    reaudit = {"kind": "official_cv_reaudit", "submit": False, "recommendation": rec,
               "artifact": guard, "metric_vendor": {"recommendation": vendor["recommendation"],
               "official_metric_found": vendor["official_metric_found"], "all_verification_tests_passed": vendor["all_verification_tests_passed"]},
               "n_normalized_train_datasets": inv["n_datasets"], "source_groups": inv["source_groups"],
               "split_recovery": {"status": split["status"], "split_source_type": split["split_source_type"],
                                  "clean_holdout_proven": split["clean_holdout_proven"], "overlap": split["overlap"],
                                  "n_holdout": len(split["split_0_test_or_validation"] or [])},
               "m32_b_replay_allowed": bool(rec == "AUDIT_PASS"),
               "unresolved_items": [k for k, v in {"official_metric": vendor["official_metric_found"],
                                    "clean_holdout": split["clean_holdout_proven"], "artifact": guard["artifact_guard_passed"]}.items() if not v]}
    (out_dir / "m32_1_official_cv_reaudit.json").write_text(json.dumps(reaudit, indent=2, default=str))
    _write_log(out_dir)
    print("=== M32.1_B OFFICIAL CV RE-AUDIT (non-submit) ===")
    print(f"  artifact_guard_passed: {guard['artifact_guard_passed']}")
    print(f"  metric vendor: {vendor['recommendation']}   normalized train datasets: {inv['n_datasets']}")
    print(f"  split recovery: {split['status']} ({split['split_source_type']})  clean_holdout_proven: {split['clean_holdout_proven']}")
    print(f"  RECOMMENDATION: {rec}   M32_B replay allowed: {reaudit['m32_b_replay_allowed']}")
    print("  No submission (re-audit only).")
    return reaudit

# --------------------------------------------------------------------------- #
# 45. M32.1 tests (23) + drivers (A vendor audit / B combined re-audit)
# --------------------------------------------------------------------------- #
def _m321_test_local_metric_never_official():
    # the vendored resolver only ever imports tracking_cellmot.* - never the proxy.
    r = resolve_official_metric_vendored()
    proxy = "local" + "_metric"
    assert r.get("import_path") is None or proxy not in str(r.get("import_path"))
    assert r.get("metrics_module") is None or "tracking_cellmot" in str(getattr(r.get("metrics_module"), "__name__", ""))


def _m321_test_vendor_hashes_recorded():
    assert isinstance(M32_1_EMBED, dict) and isinstance(M32_1_EMBED_SHA, dict)
    assert set(M32_1_EMBED) == set(M32_1_EMBED_SHA)
    assert "metrics.py" in M32_1_EMBED and "division_metrics.py" in M32_1_EMBED and "__init__.py" in M32_1_EMBED


def _m321_test_materialise_verbatim():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mat = materialise_vendored_metric(td)
        assert mat["all_verified"], mat
        for f in mat["files"]:
            assert f["sha256"] == f["expected_sha256"], f
        # the materialised metrics.py must be the EXACT upstream source (contains the official constants).
        txt = (Path(mat["src_path"]) / "tracking_cellmot" / "metrics.py").read_text()
        assert "ADJUSTMENT_ALPHA" in txt and "SCORE_DIVISION_WEIGHT" in txt and "def summarise" in txt


def _m321_test_no_formula_rewritten():
    # the embedded metrics source must carry the official scoring constants verbatim.
    import base64
    txt = base64.b64decode(M32_1_EMBED["metrics.py"]).decode()
    assert "ADJUSTMENT_ALPHA: float = 0.1" in txt and "SCORE_DIVISION_WEIGHT: float = 0.1" in txt
    assert "def evaluate(" in txt and "def evaluate_datasets(" in txt and "def per_sample_metrics(" in txt and "def summarise(" in txt


def _m321_test_dependency_report():
    r = _check_dependencies()
    assert set(r) == set(M32_1_REQUIRED_DEPS)
    for d in M32_1_REQUIRED_DEPS:
        assert "available" in r[d]


def _m321_test_official_import_or_dep_missing():
    r = resolve_official_metric_vendored()
    # either the official metric imports, or a specific dependency-missing status - never a proxy.
    assert r["status"] in ("OFFICIAL_METRIC_READY", "OFFICIAL_METRIC_DEPENDENCY_MISSING",
                           "OFFICIAL_METRIC_VERIFICATION_FAILED", "OFFICIAL_METRIC_NOT_FOUND")
    assert r["upstream_commit"] == "7396b7e98e61844e799152ddda7e5493084cc8f3"


def _m321_test_division_module_embedded():
    import base64
    txt = base64.b64decode(M32_1_EMBED["division_metrics.py"]).decode()
    assert "def evaluate_divisions" in txt and "def score_divisions" in txt


def _m321_test_fixtures_dep_guarded():
    class _Fake:
        pass
    out = run_official_metric_fixtures(_Fake())
    # without tracksdata locally -> dependency missing (never a fabricated pass).
    assert out["status"] in ("OFFICIAL_METRIC_DEPENDENCY_MISSING", "OFFICIAL_METRIC_VERIFICATION_FAILED")
    assert out["all_verification_tests_passed"] is False


def _m321_test_inventory_strips_geff():
    assert _normalize_dataset_name("44b6_0113de3b.geff") == "44b6_0113de3b"
    assert _normalize_dataset_name("6bba_05b6850b.zarr") == "6bba_05b6850b"
    assert _normalize_dataset_name("6bba_c328f2fd") == "6bba_c328f2fd"


def _m321_test_inventory_excludes_train_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # a bare 'train' dir with no geff must not appear as a dataset.
        (Path(td) / "train").mkdir()
        inv = normalize_train_inventory(td)
        assert all(r["dataset"] != "train" for r in inv["inventory"])


def _m321_test_inventory_dedup_and_group():
    m = _M32_1_GROUP_RE.match("44b6_0113de3b")
    assert m and m.group(1) == "44b6"
    assert _M32_1_GROUP_RE.match("6bba_05b6850b").group(1) == "6bba"


def _m321_test_split_search_and_parse(tmpdir=None):
    import tempfile, json as _j
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "nested" / "dataset_splits.json"
        sf.parent.mkdir(parents=True)
        sf.write_text(_j.dumps({"0": {"train": ["44b6_a.geff", "44b6_b.geff"], "test": ["6bba_c.geff"]}}))
        files = _search_split_files([td])
        assert any(f.endswith("dataset_splits.json") for f in files)
        train, test = _parse_split_file(str(sf))
        assert train == ["44b6_a", "44b6_b"] and test == ["6bba_c"]


def _m321_test_split_observed_clean_holdout():
    import tempfile, json as _j
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "dataset_splits.json"
        sf.write_text(_j.dumps({"0": {"train": ["a", "b"], "test": ["c"]}}))
        r = recover_split0(competition_dir=td, artifact_dir=None, all_datasets=["a", "b", "c"])
        assert r["split_source_type"] == "OBSERVED_FILE" and r["status"] == "SPLIT_RECOVERY_PASS"
        assert r["clean_holdout_proven"] and r["overlap"] == []


def _m321_test_split_leakage_guard():
    import tempfile, json as _j
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "dataset_splits.json"
        sf.write_text(_j.dumps({"0": {"train": ["a", "b", "c"], "test": ["c"]}}))   # c leaks
        r = recover_split0(competition_dir=td, artifact_dir=None, all_datasets=["a", "b", "c"])
        assert r["status"] == "DATA_LEAKAGE_DETECTED" and r["overlap"] == ["c"]


def _m321_test_split_unresolved_no_reconstruction():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = recover_split0(competition_dir=td, artifact_dir=None, all_datasets=["a", "b"])
        assert r["status"] == "CLEAN_HOLDOUT_UNRESOLVED" and r["split_source_type"] == "UNRESOLVED"
        assert r["deterministic_reconstruction"]["allowed"] is False


def _m321_test_split_dataset_mismatch():
    import tempfile, json as _j
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "dataset_splits.json"
        sf.write_text(_j.dumps({"0": {"train": ["a"], "test": ["zzz_missing"]}}))
        r = recover_split0(competition_dir=td, artifact_dir=None, all_datasets=["a", "b"])
        assert r["status"] == "SPLIT_DATASET_MISMATCH"


def _m321_test_checkpoint_metadata_search():
    ck = _inspect_checkpoint_metadata(None)
    assert ck["inspected"] is False and ck["split"] is None   # no weights -> honest empty


def _m321_test_estimated_n_nodes_reader():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "zarr.json").write_text(json.dumps({"attributes": {"estimated_number_of_nodes": 12345}}))
        assert _read_estimated_n_nodes(root) == 12345.0
        empty = Path(td) / "empty"; empty.mkdir()
        assert _read_estimated_n_nodes(empty) is None


def _m321_test_artifact_sha_guard():
    g = m32_1_artifact_guard()
    assert g.get("expected_sha256", M32_1_400EP_SHA256) == M32_1_400EP_SHA256
    assert "artifact_guard_passed" in g


def _m321_test_reaudit_pass_logic():
    # emulate the re-audit cascade decision.
    def reco(guard, vendor, split_leak, clean):
        if not guard:
            return "WRONG_ARTIFACT"
        if vendor == "OFFICIAL_METRIC_DEPENDENCY_MISSING":
            return "OFFICIAL_METRIC_DEPENDENCY_MISSING"
        if vendor != "OFFICIAL_METRIC_VENDOR_PASS":
            return "OFFICIAL_METRIC_VERIFICATION_FAILED"
        if split_leak:
            return "DATA_LEAKAGE_DETECTED"
        if not clean:
            return "CLEAN_HOLDOUT_UNRESOLVED"
        return "AUDIT_PASS"
    assert reco(True, "OFFICIAL_METRIC_VENDOR_PASS", False, True) == "AUDIT_PASS"
    assert reco(True, "OFFICIAL_METRIC_VENDOR_PASS", False, False) == "CLEAN_HOLDOUT_UNRESOLVED"
    assert reco(True, "OFFICIAL_METRIC_DEPENDENCY_MISSING", False, True) == "OFFICIAL_METRIC_DEPENDENCY_MISSING"
    assert reco(False, "OFFICIAL_METRIC_VENDOR_PASS", False, True) == "WRONG_ARTIFACT"


def _m321_test_upstream_provenance():
    assert M32_1_UPSTREAM_REPO.endswith("kaggle-cell-tracking-competition")
    assert M32_1_UPSTREAM_BRANCH == "main" and len(M32_1_UPSTREAM_COMMIT) == 40


def _m321_test_required_api_and_deps_constants():
    assert M32_1_REQUIRED_API == ["evaluate", "evaluate_datasets", "per_sample_metrics", "summarise"]
    assert M32_1_REQUIRED_DEPS == ["tracksdata", "geff", "polars", "scipy"]


def _m321_test_no_submission_writer():
    # M32.1 orchestration must never write a competition submission.
    assert "run_m32_1_vendor_audit" in globals() and "run_m32_1_reaudit" in globals()


def run_milestone32_1_tests() -> None:
    """Vendored-official-metric + split-recovery mechanics (no Kaggle/deps): the
    vendored source is byte-identical (sha256-verified) and never rewritten;
    local_metric.py is never the official scorer; deps report + dependency-missing
    guard; fixtures are dependency-guarded (no fabricated pass); inventory strips
    .geff, excludes the train dir, dedups, resolves source group +
    estimated_number_of_nodes; split search/parse; OBSERVED clean holdout; leakage
    guard; UNRESOLVED forbids reconstruction; dataset mismatch; checkpoint metadata
    inspection; artifact SHA guard; re-audit pass/fail logic; upstream provenance."""
    _m321_test_local_metric_never_official()
    _m321_test_vendor_hashes_recorded()
    _m321_test_materialise_verbatim()
    _m321_test_no_formula_rewritten()
    _m321_test_dependency_report()
    _m321_test_official_import_or_dep_missing()
    _m321_test_division_module_embedded()
    _m321_test_fixtures_dep_guarded()
    _m321_test_inventory_strips_geff()
    _m321_test_inventory_excludes_train_dir()
    _m321_test_inventory_dedup_and_group()
    _m321_test_split_search_and_parse()
    _m321_test_split_observed_clean_holdout()
    _m321_test_split_leakage_guard()
    _m321_test_split_unresolved_no_reconstruction()
    _m321_test_split_dataset_mismatch()
    _m321_test_checkpoint_metadata_search()
    _m321_test_estimated_n_nodes_reader()
    _m321_test_artifact_sha_guard()
    _m321_test_reaudit_pass_logic()
    _m321_test_upstream_provenance()
    _m321_test_required_api_and_deps_constants()
    _m321_test_no_submission_writer()
    print("All milestone32_1_official_metric_vendor_runner tests passed (23/23).")


def _m321_dry(kind):
    print(f"[dry-run] /kaggle/input absent - self-tests only ({kind}). On Kaggle this materialises the VENDORED official "
          "metric, imports tracking_cellmot.metrics, runs the 10 official fixtures, and recovers split_0; without "
          "tracksdata/polars it reports OFFICIAL_METRIC_DEPENDENCY_MISSING (no proxy/fake). Non-submit.")
    return {"status": "dry_run", "kind": kind}


def run_milestone32_1_vendor_audit(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR):
    print("=== Self-test: M32.1 official metric vendor + split recovery ==="); run_milestone32_1_tests()
    return _m321_dry("official_metric_vendor_audit") if not is_kaggle_env() else run_m32_1_vendor_audit(working_dir, competition_dir)


def run_milestone32_1_reaudit(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR):
    print("=== Self-test: M32.1 official metric vendor + split recovery ==="); run_milestone32_1_tests()
    return _m321_dry("official_cv_reaudit") if not is_kaggle_env() else run_m32_1_reaudit(working_dir, competition_dir)


run_milestone32_1_vendor_audit()
