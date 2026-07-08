"""
Biohub - Cell Tracking During Development
Milestone 20 (FIXED): TUNED full-chain post-processing WITH a hard baseline
guard on the pre-post-processing learned graph.

Why FIXED exists
----------------
The first M20 run was NOT submission-safe. It was meant to differ from M19-C
ONLY in the post-processing gates, but its diagnostics showed the BASE learned
graph (before any post-processing) had changed:

                       M19-C (0.880)      M20 bad run
  n_nodes_before        131797             142193
  n_edges_before        118992             127563

The predict command is byte-for-byte identical, so the base could only change
because a DIFFERENT support-pack / weights artifact was selected on Kaggle (an
alternate / TTA / different-epoch pack mounted at run time). Tuning gates on a
different base is not a controlled experiment, so that run must not be trusted.

What FIXED does
---------------
1. Reproduces M19-C's EXACT artifact/weights selection logic (same candidate
   order preferring the canonical 50ep-v1 support pack, same split_0 weights,
   same identical predict command: det 0.99, ilp-edge -1.0, appearance 0.1,
   disappearance 0.1, division 1.0, --use-ilp). It never selects an
   alternate/TTA artifact by choice.
2. Records exactly what was selected into
   /kaggle/working/milestone20_artifact_selection.json: selected artifact
   path, repo path, weights path, support-pack root, and the artifact
   manifest - so any drift is visible.
3. Adds a HARD BASELINE GUARD (tolerance 0): the pre-post-processing base must
   be n_nodes_before == 131797 AND n_edges_before == 118992. If EITHER differs,
   the report is written with baseline_guard_passed=False and
   submission_recommendation="DO_NOT_SUBMIT_BASELINE_MISMATCH", and the
   mismatch is reported loudly. Only when the base matches does it recommend
   OK_TO_SUBMIT_IF_GATES_PASS.
4. Applies ONLY the tuned post-processing gates (identical to M20): widened
   safe-division gates + caps, wider gap1 distance + caps, modest gap2
   widening (velocity direction gate held at -0.25), linefit and prune
   unchanged.

Everything else is identical to M19-C and unchanged: offline --no-deps wheel
install, sitecustomize Float16 patch, exact PYTHONPATH, dynamic LIST-based
splits, hidden-safe validation, top-level fallback, GEFF conversion basics,
submission schema, Kaggle-API behavior. Metric-validity holds (every added
edge is unit-timepoint; no direct t->t+2 edge; no motion-relink). Validation
still enforces valid, no-fallback, max_in_degree<=1, max_out_degree<=2, no
NaN, consecutive id.

IMPORTANT operational note: to reproduce the 0.880 base, attach ONLY the
canonical biohub-tracking-support-pack-50ep-v1 dataset and DETACH any
alternate / TTA / other-epoch support packs before running. The baseline guard
is the safety net if that is not done - it will refuse to recommend submission.

One primary candidate ships: VARIANT A = fullchain_tuned (guarded). Self-
contained one-cell runner. Never submits, never calls the Kaggle submission
API, never triggers Save Version. Off Kaggle it is a clean self-test-only dry
run. No prior milestone's files are modified - all reused machinery is copied
in.
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
# 21. Metric-aware post-processing of the reference GEFF graph (M20 TUNED gates)
# --------------------------------------------------------------------------- #
# All gates are physical micrometres (VOXEL_SIZE_UM / _VOXEL_SCALE come from
# the reused M16 header). Every added edge spans a UNIT timepoint (t -> t+1)
# so the evaluator can credit it; a direct t -> t+2 edge is never produced.

# Safe-division gates (geometric second-child admission).
M20_DIV_PARENT_CHILD_MAX_UM = 5.1    # source -> new (second) child
M20_DIV_SISTER_MAX_UM = 7.4         # existing child <-> new child
M20_DIV_EXISTING_CHILD_MAX_UM = 7.9  # the original single edge must not already be over-stretched
M20_DIV_FRAME_CAP_FRAC = 0.0090      # per-timepoint cap on admitted divisions
M20_DIV_GLOBAL_CAP_FRAC = 0.0052      # per-dataset cap on admitted divisions

# Single-frame gap recovery (end@t -> synthetic@t+1 -> start@t+2).
M20_GAP1_MAX_TOTAL_UM = 7.0
M20_GAP1_CAP_FRAC = 0.0075
M20_GAP1_CAP_ABS = 380

# Two-frame gap recovery (end@t -> synth@t+1 -> synth@t+2 -> start@t+3).
M20_GAP2_MAX_STEP_UM = 4.7           # per-unit-step budget (total <= 3*step by construction)
M20_GAP2_MAX_TOTAL_UM = 11.0
M20_GAP2_VELOCITY_COS_MIN = -0.25    # incoming velocity vs gap velocity must not strongly reverse
M20_GAP2_VELOCITY_NORMDIFF_UM = 6.6  # |per-step gap velocity - incoming velocity| budget
M20_GAP2_CAP_FRAC = 0.0052
M20_GAP2_CAP_ABS = 220

# Line-fit smoothing (topology-preserving coordinate blend).
M20_LINEFIT_WINDOW = 2
M20_LINEFIT_WEIGHT = 0.72

# Isolated-node pruning (degree-0 predicted nodes = single-frame tracks).
M20_PRUNE_MAX_FRAC = 0.6             # safety valve: if MORE than this fraction is isolated, skip (likely a read problem, not over-prediction)


def physical_distance_um(pos_a, pos_b) -> float:
    """Euclidean distance in micrometres between two (z, y, x) voxel-index
    positions, applying the anisotropic VOXEL_SIZE_UM scale. `_VOXEL_SCALE`
    is [z, y, x] to match the coordinate order.
    """
    d = (np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float)) * _VOXEL_SCALE
    return float(np.sqrt(float(np.dot(d, d))))


def build_base_graph(geff_graph: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduces the M16 converter's node remap + topology enforcement
    (raw -> consecutive local node_id starting at 0; drop dangling and
    backward-in-time and duplicate edges; in_degree<=1 keeping the best
    incoming; out_degree<=2 keeping the two best outgoing, ranked by higher
    edge_prob then lower edge_dist then stable order) but returns a plain
    (pred_nodes, pred_edges) pair - the SAME topology the M16 baseline
    submits. With no post-processing applied the downstream converter
    therefore reproduces the M16 baseline exactly.

    pred_nodes columns: node_id (int, >=0), t, z, y, x.
    pred_edges columns: source_id, target_id (both local node_id).
    """
    nodes = geff_graph["nodes"].copy()
    edges = geff_graph["edges"].copy()

    raw_to_local: dict[int, int] = {}
    node_rows = []
    node_t: dict[int, int] = {}
    for local_id, row in enumerate(nodes.itertuples()):
        raw_to_local[int(row.raw_id)] = local_id
        node_t[local_id] = int(row.t)
        node_rows.append({"node_id": local_id, "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
    pred_nodes = pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x"])

    kept = []
    for row in edges.itertuples():
        s_raw, t_raw = int(row.raw_source), int(row.raw_target)
        if s_raw not in raw_to_local or t_raw not in raw_to_local:
            continue
        s_kid, t_kid = raw_to_local[s_raw], raw_to_local[t_raw]
        if not (node_t[t_kid] > node_t[s_kid]):
            continue
        kept.append({"source_id": s_kid, "target_id": t_kid, "edge_prob": float(row.edge_prob), "edge_dist": float(row.edge_dist)})

    edges_df = pd.DataFrame(kept, columns=["source_id", "target_id", "edge_prob", "edge_dist"])
    if len(edges_df):
        edges_df = edges_df.drop_duplicates(subset=["source_id", "target_id"], keep="first")
        edges_df = edges_df.assign(_order=np.arange(len(edges_df)))
        edges_df = edges_df.sort_values(["edge_prob", "edge_dist", "_order"], ascending=[False, True, True]).reset_index(drop=True)
        edges_df = edges_df.drop_duplicates(subset=["target_id"], keep="first")
        edges_df = edges_df.groupby("source_id", sort=False, group_keys=False).head(2).reset_index(drop=True)

    pred_edges = edges_df[["source_id", "target_id"]].reset_index(drop=True) if len(edges_df) else pd.DataFrame(columns=["source_id", "target_id"])
    return pred_nodes, pred_edges


def _degree_maps(pred_edges: pd.DataFrame) -> tuple[dict, dict]:
    out_degree = pred_edges.groupby("source_id").size().to_dict() if len(pred_edges) else {}
    in_degree = pred_edges.groupby("target_id").size().to_dict() if len(pred_edges) else {}
    return out_degree, in_degree


def add_micro_safe_divisions(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    parent_child_max_um: float = M20_DIV_PARENT_CHILD_MAX_UM, sister_max_um: float = M20_DIV_SISTER_MAX_UM,
    existing_child_max_um: float = M20_DIV_EXISTING_CHILD_MAX_UM,
    frame_cap_frac: float = M20_DIV_FRAME_CAP_FRAC, global_cap_frac: float = M20_DIV_GLOBAL_CAP_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Admits a geometrically safe SECOND outgoing edge for a source that
    currently has EXACTLY one accepted child, turning it into a division.
    The new child must be an unmatched node (current in_degree==0) so no
    multi-parent is ever created; the source is never given a third child;
    the existing child must be exactly ONE timepoint ahead (so the sister
    edge is unit-timepoint and metric-valid) and its own edge must already
    be within `existing_child_max_um`; the new child must be within
    `parent_child_max_um` of the source and within `sister_max_um` of the
    existing child. Candidates are admitted closest-first under per-frame
    and per-dataset caps favouring the most confident (nearest) divisions.
    """
    diag = {"n_divisions_added": 0, "n_candidates_considered": 0}
    if len(pred_edges) == 0 or len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag

    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    out_degree, in_degree = _degree_maps(pred_edges)

    # sources with exactly one child whose existing edge is unit-timepoint
    # and short enough to be a credible parent.
    existing_child_of: dict[int, int] = {}
    for row in pred_edges.itertuples():
        s, t = int(row.source_id), int(row.target_id)
        if out_degree.get(s, 0) != 1:
            continue
        if t_of[t] != t_of[s] + 1:
            continue
        if physical_distance_um(pos.loc[s].to_numpy(float), pos.loc[t].to_numpy(float)) > existing_child_max_um:
            continue
        existing_child_of[s] = t

    if not existing_child_of:
        return pred_nodes.copy(), pred_edges.copy(), diag

    # orphan pool: unmatched nodes (in_degree==0) keyed by timepoint.
    orphans_by_t: dict[int, list[int]] = {}
    for nid in pred_nodes["node_id"]:
        if in_degree.get(nid, 0) == 0:
            orphans_by_t.setdefault(int(t_of[nid]), []).append(int(nid))

    candidates = []
    for s, child in existing_child_of.items():
        child_t = t_of[child]
        s_pos = pos.loc[s].to_numpy(float)
        child_pos = pos.loc[child].to_numpy(float)
        for cand in orphans_by_t.get(child_t, []):
            if cand == child or cand == s:
                continue
            c_pos = pos.loc[cand].to_numpy(float)
            d_parent = physical_distance_um(s_pos, c_pos)
            if d_parent > parent_child_max_um:
                continue
            d_sister = physical_distance_um(child_pos, c_pos)
            if d_sister > sister_max_um:
                continue
            candidates.append({"source_id": s, "new_child": cand, "t": child_t, "d_parent": d_parent})
    diag["n_candidates_considered"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag

    candidates.sort(key=lambda c: c["d_parent"])
    frame_totals: dict[int, int] = {}
    for t in t_of.values():
        frame_totals[int(t)] = frame_totals.get(int(t), 0) + 1

    # Caps limit division DENSITY but never forbid every admission outright:
    # the per-dataset global cap bounds the total (on real data int(frac*N)
    # dominates; the floor of 1 only matters on small/dry-run graphs), and
    # the per-frame cap is floored the same way so a lone confident division
    # in a small frame is still reachable while the global cap holds the line.
    global_cap = max(1, int(global_cap_frac * max(len(pred_nodes), 1)))
    frame_counts: dict[int, int] = {}
    new_edges = []
    used_children: set[int] = set()
    for c in candidates:
        if len(new_edges) >= global_cap:
            break
        s, cand, ft = c["source_id"], c["new_child"], c["t"]
        if cand in used_children:
            continue
        if in_degree.get(cand, 0) != 0:
            continue  # became matched earlier this pass (never multi-parent)
        if out_degree.get(s, 0) != 1:
            continue  # source already divided this pass (never a third child)
        frame_cap = max(1, int(frame_cap_frac * max(frame_totals.get(ft, 0), 1)))
        if frame_counts.get(ft, 0) >= frame_cap:
            continue
        new_edges.append({"source_id": s, "target_id": cand})
        out_degree[s] = out_degree.get(s, 0) + 1
        in_degree[cand] = in_degree.get(cand, 0) + 1
        frame_counts[ft] = frame_counts.get(ft, 0) + 1
        used_children.add(cand)

    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True) if new_edges else pred_edges.copy()
    diag["n_divisions_added"] = len(new_edges)
    return pred_nodes.copy(), result_edges, diag


def _ends_and_starts(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame) -> tuple[list, dict, dict]:
    """Returns tracklet END nodes (out_degree==0, not at the last
    timepoint) and START nodes (in_degree==0, not at the first timepoint)
    indexed by timepoint, plus the node->predecessor map for velocity gates.
    """
    out_degree, in_degree = _degree_maps(pred_edges)
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    max_t, min_t = int(pred_nodes["t"].max()), int(pred_nodes["t"].min())
    ends = [(int(nid), int(t_of[nid])) for nid in pred_nodes["node_id"] if out_degree.get(nid, 0) == 0 and t_of[nid] < max_t]
    starts_by_t: dict[int, list[int]] = {}
    for nid in pred_nodes["node_id"]:
        if in_degree.get(nid, 0) == 0 and t_of[nid] > min_t:
            starts_by_t.setdefault(int(t_of[nid]), []).append(int(nid))
    pred_of = dict(zip(pred_edges["target_id"], pred_edges["source_id"])) if len(pred_edges) else {}
    return ends, starts_by_t, pred_of


def close_single_frame_gaps(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    max_total_um: float = M20_GAP1_MAX_TOTAL_UM, cap_frac: float = M20_GAP1_CAP_FRAC, cap_abs: int = M20_GAP1_CAP_ABS,
    next_synth_id: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    """Recovers a single missing frame: an END node at t and a START node
    at t+2 within `max_total_um` are bridged by ONE synthetic node at t+1
    (straight-line midpoint) plus two UNIT-timepoint edges end->synth and
    synth->start. A direct end->start (t -> t+2) edge is never created.
    Synthetic node_ids are negative so they can never collide with a real
    (>=0) id; they are relabelled positive later. Each end/start is used at
    most once; admitted closest-first under a cap.
    """
    diag = {"n_gap1_candidates": 0, "n_gap1_closed": 0}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id
    ends, starts_by_t, _ = _ends_and_starts(pred_nodes, pred_edges)
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]

    candidates = []
    for end_id, end_t in ends:
        e_pos = pos.loc[end_id].to_numpy(float)
        for start_id in starts_by_t.get(end_t + 2, []):
            s_pos = pos.loc[start_id].to_numpy(float)
            total = physical_distance_um(e_pos, s_pos)
            if total <= max_total_um:
                candidates.append({"end": end_id, "start": start_id, "mid_t": end_t + 1, "total": total})
    diag["n_gap1_candidates"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id

    candidates.sort(key=lambda c: c["total"])
    cap = min(cap_abs, max(1, int(cap_frac * max(len(pred_nodes), 1))))
    new_nodes, new_edges = [], []
    used_ends, used_starts = set(), set()
    sid = next_synth_id
    for c in candidates:
        if len(new_nodes) >= cap:
            break
        if c["end"] in used_ends or c["start"] in used_starts:
            continue
        e_pos = pos.loc[c["end"]].to_numpy(float)
        s_pos = pos.loc[c["start"]].to_numpy(float)
        mid = (e_pos + s_pos) / 2.0
        new_nodes.append({"node_id": sid, "t": c["mid_t"], "z": float(mid[0]), "y": float(mid[1]), "x": float(mid[2])})
        new_edges.append({"source_id": c["end"], "target_id": sid})
        new_edges.append({"source_id": sid, "target_id": c["start"]})
        used_ends.add(c["end"])
        used_starts.add(c["start"])
        sid -= 1

    result_nodes = pd.concat([pred_nodes, pd.DataFrame(new_nodes, columns=["node_id", "t", "z", "y", "x"])], ignore_index=True) if new_nodes else pred_nodes.copy()
    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True) if new_edges else pred_edges.copy()
    diag["n_gap1_closed"] = len(new_nodes)
    return result_nodes, result_edges, diag, sid


def recover_two_frame_gaps(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    max_step_um: float = M20_GAP2_MAX_STEP_UM, max_total_um: float = M20_GAP2_MAX_TOTAL_UM,
    velocity_cos_min: float = M20_GAP2_VELOCITY_COS_MIN, velocity_normdiff_um: float = M20_GAP2_VELOCITY_NORMDIFF_UM,
    cap_frac: float = M20_GAP2_CAP_FRAC, cap_abs: int = M20_GAP2_CAP_ABS, next_synth_id: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    """Recovers two missing frames: an END node at t and a START node at
    t+3 are bridged by TWO synthetic nodes at t+1 and t+2 (linear
    interpolation) plus three UNIT-timepoint edges. Gated on per-step and
    total physical distance AND on velocity consistency - when the end node
    has a predecessor, the incoming velocity and the gap velocity must not
    strongly reverse (cosine >= `velocity_cos_min`) and their per-step
    magnitudes must agree within `velocity_normdiff_um`. No direct multi-
    frame edge is ever created. Synthetic ids are negative; admitted
    closest-first under a cap.
    """
    diag = {"n_gap2_candidates": 0, "n_gap2_recovered": 0}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id
    ends, starts_by_t, pred_of = _ends_and_starts(pred_nodes, pred_edges)
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))

    def _scaled(nid):
        return pos.loc[nid].to_numpy(float) * _VOXEL_SCALE

    candidates = []
    for end_id, end_t in ends:
        e_um = _scaled(end_id)
        # incoming velocity (per unit step) if the end has a predecessor.
        v_in = None
        if end_id in pred_of:
            p = pred_of[end_id]
            dt = int(t_of[end_id]) - int(t_of[p])
            if dt > 0:
                v_in = (e_um - _scaled(p)) / float(dt)
        for start_id in starts_by_t.get(end_t + 3, []):
            s_um = _scaled(start_id)
            total = float(np.linalg.norm(s_um - e_um))
            if total > max_total_um:
                continue
            v_gap = (s_um - e_um) / 3.0
            if float(np.linalg.norm(v_gap)) > max_step_um:
                continue
            if v_in is not None:
                nin, ngap = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_gap))
                if nin > 1e-9 and ngap > 1e-9:
                    cos = float(np.dot(v_in, v_gap) / (nin * ngap))
                    if cos < velocity_cos_min:
                        continue
                    if float(np.linalg.norm(v_gap - v_in)) > velocity_normdiff_um:
                        continue
            candidates.append({"end": end_id, "start": start_id, "t0": end_t, "total": total})
    diag["n_gap2_candidates"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id

    candidates.sort(key=lambda c: c["total"])
    cap = min(cap_abs, max(1, int(cap_frac * max(len(pred_nodes), 1))))
    new_nodes, new_edges = [], []
    used_ends, used_starts = set(), set()
    sid = next_synth_id
    for c in candidates:
        if len(new_nodes) >= 2 * cap:  # each recovery adds two nodes
            break
        if c["end"] in used_ends or c["start"] in used_starts:
            continue
        e_pos = pos.loc[c["end"]].to_numpy(float)
        s_pos = pos.loc[c["start"]].to_numpy(float)
        m1 = e_pos + (s_pos - e_pos) * (1.0 / 3.0)
        m2 = e_pos + (s_pos - e_pos) * (2.0 / 3.0)
        id1, id2 = sid, sid - 1
        new_nodes.append({"node_id": id1, "t": c["t0"] + 1, "z": float(m1[0]), "y": float(m1[1]), "x": float(m1[2])})
        new_nodes.append({"node_id": id2, "t": c["t0"] + 2, "z": float(m2[0]), "y": float(m2[1]), "x": float(m2[2])})
        new_edges.append({"source_id": c["end"], "target_id": id1})
        new_edges.append({"source_id": id1, "target_id": id2})
        new_edges.append({"source_id": id2, "target_id": c["start"]})
        used_ends.add(c["end"])
        used_starts.add(c["start"])
        sid -= 2

    result_nodes = pd.concat([pred_nodes, pd.DataFrame(new_nodes, columns=["node_id", "t", "z", "y", "x"])], ignore_index=True) if new_nodes else pred_nodes.copy()
    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])], ignore_index=True) if new_edges else pred_edges.copy()
    diag["n_gap2_recovered"] = len(new_nodes) // 2
    return result_nodes, result_edges, diag, sid


def linefit_smooth_nodes(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    window: int = M20_LINEFIT_WINDOW, weight: float = M20_LINEFIT_WEIGHT,
) -> tuple[pd.DataFrame, dict]:
    """Topology-preserving coordinate smoothing - never adds/removes a node
    or edge, only reassigns (z, y, x) for "linear interior" nodes (exactly
    one predecessor at t-1 AND exactly one successor at t+1). Division
    nodes, branch points, and chain endpoints are left untouched. Blends
    weight*original + (1-weight)*mean(up-to-`window` neighbours each side).
    """
    diag = {"n_nodes_smoothed": 0}
    if len(pred_nodes) == 0 or len(pred_edges) == 0:
        return pred_nodes.copy(), diag

    succ = dict(zip(pred_edges["source_id"], pred_edges["target_id"]))
    pred = dict(zip(pred_edges["target_id"], pred_edges["source_id"]))
    out_degree, in_degree = _degree_maps(pred_edges)
    node_lookup = pred_nodes.set_index("node_id")
    t_lookup = node_lookup["t"].to_dict()

    def is_linear_interior(nid) -> bool:
        if out_degree.get(nid, 0) != 1 or in_degree.get(nid, 0) != 1:
            return False
        s_id, p_id = succ.get(nid), pred.get(nid)
        if s_id is None or p_id is None:
            return False
        return t_lookup.get(s_id) == t_lookup[nid] + 1 and t_lookup.get(p_id) == t_lookup[nid] - 1

    def collect(nid, direction, steps):
        out = []
        cur = nid
        for _ in range(steps):
            nxt = succ.get(cur) if direction == "succ" else pred.get(cur)
            if nxt is None:
                break
            out.append(nxt)
            if not is_linear_interior(nxt):
                break
            cur = nxt
        return out

    smoothed: dict = {}
    for nid in pred_nodes["node_id"]:
        if not is_linear_interior(nid):
            continue
        neigh = collect(nid, "succ", window) + collect(nid, "pred", window)
        if not neigh:
            continue
        neigh_avg = node_lookup.loc[neigh, ["z", "y", "x"]].to_numpy(float).mean(axis=0)
        original = node_lookup.loc[nid, ["z", "y", "x"]].to_numpy(float)
        smoothed[nid] = weight * original + (1.0 - weight) * neigh_avg

    result = pred_nodes.copy()
    if smoothed:
        idx = result.set_index("node_id")
        for nid, p in smoothed.items():
            idx.loc[nid, ["z", "y", "x"]] = p
        result = idx.reset_index()[["node_id", "t", "z", "y", "x"]]
    diag["n_nodes_smoothed"] = len(smoothed)
    return result, diag


def prune_isolated_nodes(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, max_frac: float = M20_PRUNE_MAX_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Removes degree-0 nodes (no incoming AND no outgoing edge) - the
    single-frame tracks that inflate the predicted node count T_pred and
    are penalised by the metric with zero edge benefit. Removing them can
    never lower edge_jaccard (they carry no edges) or division_jaccard
    (out_degree 0 is not a division). A safety valve skips pruning entirely
    if MORE than `max_frac` of nodes are isolated, since that signals a
    graph-read problem rather than genuine over-prediction (better to keep
    the M16 baseline than gut it).
    """
    diag = {"n_isolated": 0, "n_pruned": 0, "prune_skipped_reason": None}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag
    matched = set(pred_edges["source_id"]) | set(pred_edges["target_id"]) if len(pred_edges) else set()
    isolated_mask = ~pred_nodes["node_id"].isin(matched)
    n_isolated = int(isolated_mask.sum())
    diag["n_isolated"] = n_isolated
    if n_isolated == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag
    if n_isolated > max_frac * len(pred_nodes):
        diag["prune_skipped_reason"] = f"{n_isolated}/{len(pred_nodes)} isolated exceeds max_frac={max_frac}; kept baseline"
        return pred_nodes.copy(), pred_edges.copy(), diag
    result_nodes = pred_nodes[~isolated_mask].reset_index(drop=True)
    diag["n_pruned"] = n_isolated
    return result_nodes, pred_edges.copy(), diag


def relabel_positive(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, start_id: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Remaps every node_id (real >=0 AND synthetic <0) to positive
    consecutive ids beginning at `start_id`, in node-row order, and rewrites
    edges through the same map (dropping any edge whose endpoint no longer
    exists, e.g. after isolated pruning). Guarantees no negative id ever
    reaches the submission. Returns (nodes, edges, next_start).
    """
    id_map: dict[int, int] = {}
    rows = []
    cur = start_id
    for row in pred_nodes.itertuples():
        id_map[int(row.node_id)] = cur
        rows.append({"node_id": cur, "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)})
        cur += 1
    nodes_out = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])

    e_rows = []
    for row in pred_edges.itertuples() if len(pred_edges) else []:
        s, t = int(row.source_id), int(row.target_id)
        if s in id_map and t in id_map:
            e_rows.append({"source_id": id_map[s], "target_id": id_map[t]})
    edges_out = pd.DataFrame(e_rows, columns=["source_id", "target_id"])
    return nodes_out, edges_out, cur


# Ordered post-processing operation registry. Each op takes/returns
# (pred_nodes, pred_edges) plus a running synthetic-id counter where needed.
def apply_postprocess_chain(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, ops: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Runs the requested ordered ops. Divisions run before gaps so an
    orphan can be rescued as a sister child before it is considered a gap
    start; smoothing runs after structure is final; isolated pruning runs
    last so divisions/gaps get first chance to rescue orphans. Degrees are
    recomputed inside each op, so ops compose safely.
    """
    diag: dict = {"ops": list(ops)}
    synth = -1
    for op in ops:
        if op == "safe_divisions":
            pred_nodes, pred_edges, d = add_micro_safe_divisions(pred_nodes, pred_edges)
            diag["safe_divisions"] = d
        elif op == "gap1":
            pred_nodes, pred_edges, d, synth = close_single_frame_gaps(pred_nodes, pred_edges, next_synth_id=synth)
            diag["gap1"] = d
        elif op == "gap2":
            pred_nodes, pred_edges, d, synth = recover_two_frame_gaps(pred_nodes, pred_edges, next_synth_id=synth)
            diag["gap2"] = d
        elif op == "linefit":
            pred_nodes, d = linefit_smooth_nodes(pred_nodes, pred_edges)
            diag["linefit"] = d
        elif op == "prune_isolated":
            pred_nodes, pred_edges, d = prune_isolated_nodes(pred_nodes, pred_edges)
            diag["prune_isolated"] = d
        else:
            raise ValueError(f"unknown postprocess op {op!r}")
    return pred_nodes, pred_edges, diag


def convert_predictions_to_submission_postprocessed(
    predictions_dir: Path, expected_datasets: Sequence[str], ops: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    """Per-dataset: read GEFF -> build_base_graph (M16-identical topology)
    -> apply_postprocess_chain(ops) -> relabel to globally-consecutive
    positive node_ids -> emit node rows then edge rows with a global running
    `id`. With `ops` empty the output reproduces the M16 baseline. Reports
    before/after node & edge counts and per-dataset postprocess diagnostics.
    """
    stores = find_prediction_geff_stores(predictions_dir)
    all_rows = []
    per_dataset = []
    next_id = 0
    next_node_id = 0
    tot_nodes_before = tot_nodes_after = tot_edges_before = tot_edges_after = 0
    for dataset_name, geff_root in stores:
        geff_graph = read_geff_graph(geff_root)
        base_nodes, base_edges = build_base_graph(geff_graph)
        n_nodes_before, n_edges_before = len(base_nodes), len(base_edges)
        pp_nodes, pp_edges, pp_diag = apply_postprocess_chain(base_nodes, base_edges, ops)
        pp_nodes, pp_edges, next_node_id = relabel_positive(pp_nodes, pp_edges, next_node_id)

        for row in pp_nodes.itertuples():
            all_rows.append({"id": next_id, "dataset": dataset_name, "row_type": "node", "node_id": int(row.node_id),
                             "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x), "source_id": -1, "target_id": -1})
            next_id += 1
        for row in pp_edges.itertuples() if len(pp_edges) else []:
            all_rows.append({"id": next_id, "dataset": dataset_name, "row_type": "edge", "node_id": -1,
                             "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(row.source_id), "target_id": int(row.target_id)})
            next_id += 1

        tot_nodes_before += n_nodes_before
        tot_nodes_after += len(pp_nodes)
        tot_edges_before += n_edges_before
        tot_edges_after += len(pp_edges)
        per_dataset.append({"dataset": dataset_name, "n_nodes_before": n_nodes_before, "n_nodes_after": len(pp_nodes),
                            "n_edges_before": n_edges_before, "n_edges_after": len(pp_edges), "postprocess": pp_diag})
        print(f"  [pp] {dataset_name}: nodes {n_nodes_before}->{len(pp_nodes)} edges {n_edges_before}->{len(pp_edges)} ops={list(ops)}")

    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    conversion = {
        "predictions_dir": str(predictions_dir), "n_stores_found": len(stores),
        "found_datasets": [n for n, _ in stores], "expected_datasets": list(expected_datasets), "ops": list(ops),
        "n_nodes_before": tot_nodes_before, "n_nodes_after": tot_nodes_after,
        "n_edges_before": tot_edges_before, "n_edges_after": tot_edges_after, "per_dataset": per_dataset,
    }
    return submission_df, conversion

# --------------------------------------------------------------------------- #
# 22. Milestone 20 variant registry + command builder (single tuned full_chain)
# --------------------------------------------------------------------------- #
# ONE primary candidate. The predict command is the EXACT M16 command (det
# 0.99 + reference ILP weights); only the post-processing GATES differ from
# M19-C (see section 21). The op ORDER is identical to M19-C's full_chain.
M20_VARIANTS = {
    "A": {"label": "fullchain_tuned", "ops": ["safe_divisions", "gap1", "gap2", "linefit", "prune_isolated"]},
}


def build_m20_command(
    variant_name: str, introspection: dict,
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
) -> tuple[list[str], dict]:
    """Builds the predict command - IDENTICAL to M16/M19-C: det 0.99 +
    reference ILP weights, all confirmed-accepted flags, nothing to drop.
    The tuned post-processing chain is applied AFTER prediction, never as a
    flag.
    """
    params = dict(BASE_PREDICT_PARAMS)  # det 0.99 + reference ILP weights, unchanged
    available_splits = list(introspection.get("weight_splits", [])) or [0]
    split = select_weight_split(0, available_splits)
    cmd = build_base_command(params, split, splits_filename, data_dir)
    vdef = M20_VARIANTS[variant_name]
    notes = {
        "variant": variant_name, "label": vdef["label"], "ops": vdef["ops"],
        "det_threshold": params["det_threshold"], "split_used": split, "available_splits": available_splits,
        "predict_command_identical_to_m16": True, "gates": "tuned_vs_m19c",
    }
    return cmd, notes

# --------------------------------------------------------------------------- #
# 23. M20 (FIXED) orchestration + HARD BASELINE GUARD + top-level fallback
# --------------------------------------------------------------------------- #
# The base learned graph (BEFORE post-processing) MUST match M19-C's exactly,
# or the run is not a controlled gate-tuning experiment and must not be
# submitted. Tolerance is ZERO.
EXPECTED_N_NODES_BEFORE = 131797   # M19-C (public 0.880) pre-postprocess node count
EXPECTED_N_EDGES_BEFORE = 118992   # M19-C (public 0.880) pre-postprocess edge count
BASELINE_TOLERANCE = 0


def write_artifact_selection(out_dir: Path, artifact: dict, materialize: dict, weights_rel: str) -> dict:
    """Records EXACTLY which artifact/repo/weights/support-pack were selected,
    plus the artifact manifest, into milestone20_artifact_selection.json - so
    any base-graph drift (a different pack mounted) is diagnosable. Does not
    change selection; only reports it.
    """
    artifact_dir = artifact.get("artifact_dir")
    manifest = None
    if artifact_dir:
        manifest_path = Path(artifact_dir) / "ARTIFACT_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception as exc:
                manifest = {"_manifest_read_error": repr(exc)}
    record = {
        "resolved": artifact.get("resolved"),
        "support_pack_root": artifact_dir,
        "selected_artifact_path": artifact_dir,
        "repo_path": materialize.get("repo_dst") if materialize else None,
        "weights_path": (str(Path(materialize["repo_dst"]) / weights_rel) if materialize else None),
        "weights_mode": materialize.get("weights_mode") if materialize else None,
        "artifact_manifest": manifest,
        "candidate_dirs_in_preference_order": list(ARTIFACT_CANDIDATE_DIRS),
        "checked_candidates": artifact.get("checked"),
    }
    (out_dir / "milestone20_artifact_selection.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def check_baseline_guard(conversion: dict) -> dict:
    """Hard guard (tolerance 0): the pre-post-processing base must equal
    M19-C's n_nodes_before / n_edges_before. If EITHER differs, recommend
    DO_NOT_SUBMIT_BASELINE_MISMATCH. A missing count also fails closed.
    """
    actual_nodes = conversion.get("n_nodes_before")
    actual_edges = conversion.get("n_edges_before")
    nodes_ok = actual_nodes is not None and abs(int(actual_nodes) - EXPECTED_N_NODES_BEFORE) <= BASELINE_TOLERANCE
    edges_ok = actual_edges is not None and abs(int(actual_edges) - EXPECTED_N_EDGES_BEFORE) <= BASELINE_TOLERANCE
    passed = bool(nodes_ok and edges_ok)
    detail = None
    if not passed:
        detail = (
            f"BASE LEARNED GRAPH CHANGED vs M19-C: n_nodes_before={actual_nodes} "
            f"(expected {EXPECTED_N_NODES_BEFORE}), n_edges_before={actual_edges} "
            f"(expected {EXPECTED_N_EDGES_BEFORE}). A different support-pack/weights artifact was "
            f"selected. Attach ONLY the canonical biohub-tracking-support-pack-50ep-v1 dataset and "
            f"detach any alternate/TTA/other-epoch packs, then re-run. DO NOT SUBMIT this output."
        )
    return {
        "baseline_guard_passed": passed,
        "expected_n_nodes_before": EXPECTED_N_NODES_BEFORE,
        "expected_n_edges_before": EXPECTED_N_EDGES_BEFORE,
        "actual_n_nodes_before": actual_nodes,
        "actual_n_edges_before": actual_edges,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "submission_recommendation": "OK_TO_SUBMIT_IF_GATES_PASS" if passed else "DO_NOT_SUBMIT_BASELINE_MISMATCH",
        "mismatch_detail": detail,
    }


def run_m20_variant_pipeline(
    variant_name: str, working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
) -> dict:
    """One M20 (FIXED) variant end to end, reusing every M16/M17 step
    (resolve -> materialize -> sitecustomize -> offline install -> import
    check -> introspection -> DYNAMIC list-splits -> the EXACT M16 predict
    command) then applying the tuned post-processing chain, VALIDATING, and -
    critically - running the HARD BASELINE GUARD on the pre-post-processing
    graph before recommending submission. Raises on any fatal step so the
    top-level fallback guarantees a valid submission.csv.
    """
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = Path(competition_dir)
    test_dir = comp / "test"
    sample_submission_path = comp / "sample_submission.csv"
    vdef = M20_VARIANTS[variant_name]

    log(f"=== Milestone 20 (FIXED) variant {variant_name} ({vdef['label']}) ops={vdef['ops']} ===")

    artifact = resolve_artifact()
    if not artifact["resolved"]:
        # Still record what was checked so the mount problem is diagnosable.
        write_artifact_selection(out_dir, artifact, {}, weights_rel_for_split(0))
        raise RuntimeError("support-pack artifact not found at any candidate mount (see milestone20_artifact_selection.json)")
    artifact_dir = Path(artifact["artifact_dir"])
    log(f"Artifact resolved: {artifact_dir}")

    materialize = materialize_repo(artifact_dir, out_dir)
    log(f"Repo materialized to {materialize['repo_dst']} (weights: {materialize['weights_mode']})")
    write_sitecustomize()
    env = build_subprocess_env()

    introspection = introspect_artifact(artifact_dir, out_dir)

    command, cmd_notes = build_m20_command(variant_name, introspection)
    # Record the exact artifact/repo/weights/support-pack selection now that
    # the split (hence weights path) is known.
    selection = write_artifact_selection(out_dir, artifact, materialize, weights_rel_for_split(cmd_notes["split_used"]))
    log(f"Artifact selection recorded: support_pack_root={selection['support_pack_root']} weights_path={selection['weights_path']}")

    dep_report = install_dependencies(artifact_dir / "wheels", env)
    (out_dir / "milestone20_dependency_report.json").write_text(json.dumps(dep_report, indent=2))
    if not dep_report["all_ok"]:
        raise RuntimeError("offline wheel install failed (see milestone20_dependency_report.json)")

    import_report = verify_imports(env)
    if not import_report["all_ok"]:
        raise RuntimeError(f"post-install import check failed: {import_report['modules']}")
    log(f"Imports OK: {list(import_report['modules'].keys())}")

    repo_dst = Path(materialize["repo_dst"])
    shipped_template = repo_dst / SPLITS_FILENAME
    splits_result = prepare_splits(
        test_dir=test_dir, sample_submission_path=sample_submission_path,
        template_path=shipped_template if shipped_template.exists() else None,
        out_splits_path=repo_dst / SPLITS_FILENAME, diagnostic_path=out_dir / "milestone20_splits.json",
    )

    (out_dir / "milestone20_run_command.json").write_text(json.dumps({"variant": variant_name, "command": command, "cwd": str(repo_dst), "notes": cmd_notes}, indent=2, default=str))
    log(f"Variant {variant_name} command (== M16): {' '.join(command)}")

    exec_result = run_prediction(command, cwd=repo_dst, env=env)
    (out_dir / "milestone20_execute_result.json").write_text(json.dumps(exec_result, indent=2))
    if not exec_result["ok"]:
        raise RuntimeError("predict script returned nonzero (see milestone20_execute_result.json)")

    predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / f"split_{cmd_notes['split_used']}"
    if not predictions_dir.exists():
        predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / "split_0"
    expected_datasets = discover_test_datasets(test_dir, sample_submission_path)["chosen"]

    submission_df, conversion = convert_predictions_to_submission_postprocessed(
        predictions_dir, expected_datasets, ops=vdef["ops"])
    (out_dir / "milestone20_geff_conversion.json").write_text(json.dumps(conversion, indent=2, default=str))
    log(f"  [pp] ops={vdef['ops']} nodes {conversion['n_nodes_before']}->{conversion['n_nodes_after']} "
        f"edges {conversion['n_edges_before']}->{conversion['n_edges_after']}")

    # --- HARD BASELINE GUARD (the fix): base must match M19-C exactly. ---
    guard = check_baseline_guard(conversion)
    if guard["baseline_guard_passed"]:
        log(f"[baseline-guard] PASSED: base matches M19-C "
            f"(nodes={guard['actual_n_nodes_before']}, edges={guard['actual_n_edges_before']}).")
    else:
        log("[baseline-guard] *** FAILED *** " + guard["mismatch_detail"])

    valid, report, stats = validate_reference_submission(submission_df, expected_datasets)
    log(report)
    if not valid:
        raise RuntimeError("post-processed submission FAILED validation (falling back)")

    submission_path = out_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    submission_df.to_csv(out_dir / f"submission_m20_variant_{variant_name}.csv", index=False)
    write_reference_diagnostics(submission_df, valid, stats, out_dir)

    n_nodes = int((submission_df["row_type"] == "node").sum())
    n_edges = int((submission_df["row_type"] == "edge").sum())
    submission_recommendation = guard["submission_recommendation"]
    report_obj = {
        "variant": variant_name, "label": vdef["label"], "ops": vdef["ops"], "command": command,
        "command_notes": cmd_notes, "postprocess": conversion, "artifact_selection": selection,
        "baseline_guard": guard, "baseline_guard_passed": guard["baseline_guard_passed"],
        "submission_recommendation": submission_recommendation,
        "expected_datasets": expected_datasets, "split0_test_names": splits_result["split0_test_names"],
        "submission_shape": list(submission_df.shape), "n_node_rows": n_nodes, "n_edge_rows": n_edges,
        "max_in_degree": stats["max_in_degree"], "max_out_degree": stats["max_out_degree"],
        "valid": valid, "fallback_used": False, "final_source": "reference_learned_graph_postprocessed",
        "final_instruction": (
            "submission.csv built and validated. " + (
                "Baseline guard PASSED - OK to submit only after reviewing the on-Kaggle gates."
                if guard["baseline_guard_passed"] else
                "BASELINE GUARD FAILED - DO NOT SUBMIT (base learned graph changed vs M19-C).")),
    }
    (out_dir / "milestone20_reference_submission_report.json").write_text(json.dumps(report_obj, indent=2, default=str))

    print("\n=== Variant summary ===")
    print(f"  variant: {variant_name} ({vdef['label']})  ops: {vdef['ops']}")
    print(f"  final_source: reference_learned_graph_postprocessed")
    print(f"  fallback_used: False")
    print(f"  valid: {valid}")
    print(f"  submission shape: {submission_df.shape}")
    print(f"  node rows: {n_nodes}  edge rows: {n_edges}")
    print(f"  postprocess: nodes {conversion['n_nodes_before']}->{conversion['n_nodes_after']} "
          f"edges {conversion['n_edges_before']}->{conversion['n_edges_after']}")
    print(f"  topology: max_in_degree={stats['max_in_degree']} max_out_degree={stats['max_out_degree']} "
          f"dangling={stats['dangling']} duplicate={stats['duplicate']}")
    print(f"  BASELINE GUARD: passed={guard['baseline_guard_passed']} "
          f"(base nodes {guard['actual_n_nodes_before']}/expected {guard['expected_n_nodes_before']}, "
          f"edges {guard['actual_n_edges_before']}/expected {guard['expected_n_edges_before']})")
    print(f"  SUBMISSION RECOMMENDATION: {submission_recommendation}")
    if not guard["baseline_guard_passed"]:
        print("  *** " + guard["mismatch_detail"])
    log("submission.csv built and validated. Do not submit until user reviews (and only if baseline guard passed).")
    return {"submission_df": submission_df, "variant": variant_name, "valid": valid,
            "baseline_guard_passed": guard["baseline_guard_passed"], "report": report_obj}


def run_m20_variant_with_fallback(
    variant_name: str, working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
) -> dict:
    """Runs one variant; on ANY failure, if no submission.csv exists, writes a
    valid fallback from sample_submission.csv. Re-raises ONLY if even the
    fallback cannot be written. Identical safety contract to M16-M19.
    """
    _RUN_LOG.clear()
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    sample_submission_path = Path(competition_dir) / "sample_submission.csv"
    try:
        result = run_m20_variant_pipeline(variant_name, working_dir, competition_dir)
        _write_log(out_dir)
        return {"status": "ok", **result}
    except Exception:
        tb = traceback.format_exc()
        log("[error] variant pipeline failed:\n" + tb)
        if submission_path.exists():
            log("submission.csv already exists; keeping it (not overwriting with fallback).")
            _write_log(out_dir)
            return {"status": "pipeline_failed_submission_exists"}
        try:
            fb = write_fallback_submission(sample_submission_path, submission_path)
            (out_dir / "milestone20_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback": fb}, indent=2))
            print("\n=== Variant summary ===")
            print(f"  variant: {variant_name}")
            print(f"  final_source: fallback_sample_submission")
            print(f"  fallback_used: True")
            print(f"  SUBMISSION RECOMMENDATION: DO_NOT_SUBMIT_FALLBACK")
            log("FALLBACK sample submission written because learned graph pipeline failed.")
            _write_log(out_dir)
            return {"status": "fallback_written", "fallback": fb}
        except Exception:
            fb_tb = traceback.format_exc()
            try:
                (out_dir / "milestone20_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback_error": fb_tb}, indent=2))
            except Exception:
                pass
            log("[fatal] could not write fallback submission either:\n" + fb_tb)
            _write_log(out_dir)
            raise

# --------------------------------------------------------------------------- #
# 24. Tests
# --------------------------------------------------------------------------- #
def _write_synthetic_geff_store(geff_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame,
                                node_solution=None, edge_solution=None, edge_prob=None, edge_dist=None) -> bool:
    """Writes a synthetic GEFF zarr-v3 store using the real `zarr` library
    (so the manual reader is exercised end-to-end). Returns False if zarr
    isn't importable, so the caller can skip the roundtrip sub-test rather
    than fail on an environment without zarr.
    """
    try:
        import zarr
        from zarr.codecs import BytesCodec, ZstdCodec
    except Exception:
        return False
    if geff_dir.exists():
        shutil.rmtree(geff_dir)
    store = zarr.storage.LocalStore(str(geff_dir))
    g = zarr.open_group(store=store, mode="w", zarr_format=3)

    def arr(path, data):
        data = np.asarray(data)
        a = g.create_array(path, shape=data.shape, chunks=data.shape, dtype=data.dtype,
                           serializer=BytesCodec(endian="little"), compressors=[ZstdCodec(level=3)])
        a[:] = data

    arr("nodes/ids", nodes["raw_id"].to_numpy(np.int64))
    arr("nodes/props/t/values", nodes["t"].to_numpy(np.int64))
    arr("nodes/props/z/values", nodes["z"].to_numpy(np.float32))
    arr("nodes/props/y/values", nodes["y"].to_numpy(np.float32))
    arr("nodes/props/x/values", nodes["x"].to_numpy(np.float32))
    if node_solution is not None:
        arr("nodes/props/solution/values", np.asarray(node_solution, dtype=np.int64))
    arr("edges/ids", edges[["raw_source", "raw_target"]].to_numpy(np.int64))
    if edge_solution is not None:
        arr("edges/props/solution/values", np.asarray(edge_solution, dtype=np.int64))
    if edge_prob is not None:
        arr("edges/props/edge_prob/values", np.asarray(edge_prob, dtype=np.float32))
    if edge_dist is not None:
        arr("edges/props/edge_dist/values", np.asarray(edge_dist, dtype=np.float32))
    return True


def _assert_edges_unit_t(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, ctx: str) -> None:
    """The metric credits ONLY unit-timepoint edges: assert every edge spans
    exactly one timepoint (t_target - t_source == 1). A t->t+2 (or longer)
    edge would be metric-invalid; this is the hard gate for gap recovery.
    """
    if len(pred_edges) == 0:
        return
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    for row in pred_edges.itertuples():
        dt = int(t_of[int(row.target_id)]) - int(t_of[int(row.source_id)])
        assert dt == 1, f"{ctx}: non-unit edge {row.source_id}->{row.target_id} spans {dt} timepoints"


def _assert_topology_ok(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, ctx: str) -> None:
    """No dangling edge, no multi-parent (in_degree>1), out_degree<=2."""
    ids = set(pred_nodes["node_id"])
    if len(pred_edges) == 0:
        return
    for row in pred_edges.itertuples():
        assert int(row.source_id) in ids and int(row.target_id) in ids, f"{ctx}: dangling edge"
    in_deg = pred_edges.groupby("target_id").size()
    out_deg = pred_edges.groupby("source_id").size()
    assert int(in_deg.max()) <= 1, f"{ctx}: multi-parent present"
    assert int(out_deg.max()) <= 2, f"{ctx}: out_degree>2 present"


def run_milestone20_tests() -> None:
    """Unit tests for M20 metric-aware post-processing on temp/in-memory
    fixtures - no Kaggle, no artifact, no GPU. Covers: base-graph topology
    enforcement, safe-division gates (unit-t existing child, no multi-parent,
    no third child, non-unit existing child rejected), single/two-frame gap
    recovery producing ONLY unit-timepoint edges (never a direct multi-frame
    edge), velocity gating, isolated-node pruning + safety valve, positive
    relabelling, and an end-to-end GEFF roundtrip whose empty-ops output
    equals the M16 baseline and whose full chain stays valid.
    """
    scratch = Path("/tmp") / "m19_test_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # Test 1: build_base_graph reproduces M16 topology enforcement. Node 3
    # has two incoming candidates (from 0 prob .9, from 1 prob .5); in<=1
    # keeps only the higher-prob one. A backward-in-time edge is dropped.
    geff = {
        "nodes": pd.DataFrame({"raw_id": [10, 11, 12, 13], "t": [0, 0, 0, 1],
                               "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 3.0], "x": [0.0] * 4}),
        "edges": pd.DataFrame({"raw_source": [10, 11, 13], "raw_target": [13, 13, 12],
                               "edge_prob": [0.9, 0.5, 0.9], "edge_dist": [1.0, 1.0, 1.0]}),
        "has_edge_prob": True, "has_edge_dist": True,
    }
    bn, be = build_base_graph(geff)
    assert len(bn) == 4 and list(bn["node_id"]) == [0, 1, 2, 3], "nodes remap to consecutive 0..n"
    assert len(be) == 1, "in<=1 keeps a single incoming edge to node 3; backward edge 13->12 dropped"
    assert (int(be.iloc[0]["source_id"]), int(be.iloc[0]["target_id"])) == (0, 3)

    # Test 2: safe division admits ONE geometrically valid sister. Source 0
    # @t0 has one child 1 @t1 (unit ahead); orphan 2 @t1 sits within all
    # gates -> becomes the second child. Voxel y-scale 0.40625um: y=1 -> ~0.41um.
    nodes = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 1, 1], "z": [0.0] * 3, "y": [0.0, 1.0, 2.0], "x": [0.0] * 3})
    edges = pd.DataFrame({"source_id": [0], "target_id": [1]})
    n2, e2, d2 = add_micro_safe_divisions(nodes, edges)
    assert d2["n_divisions_added"] == 1, "one valid division admitted"
    assert set(zip(e2["source_id"], e2["target_id"])) == {(0, 1), (0, 2)}
    _assert_topology_ok(n2, e2, "safe_division"); _assert_edges_unit_t(n2, e2, "safe_division")

    # Test 2b: non-unit existing child is rejected (child at t2, not t1) - a
    # sister would create a t0->t2 metric-invalid edge, so no division.
    nodes_nu = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 2, 2], "z": [0.0] * 3, "y": [0.0, 1.0, 2.0], "x": [0.0] * 3})
    _, e2b, d2b = add_micro_safe_divisions(nodes_nu, pd.DataFrame({"source_id": [0], "target_id": [1]}))
    assert d2b["n_divisions_added"] == 0, "existing child not one frame ahead -> no division"

    # Test 2c: multi-parent guard - the orphan already has a parent, so it is
    # never given a second one.
    nodes_mp = pd.DataFrame({"node_id": [0, 1, 2, 3], "t": [0, 1, 1, 0], "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 2.0], "x": [0.0] * 4})
    edges_mp = pd.DataFrame({"source_id": [0, 3], "target_id": [1, 2]})  # node 2 already parented by 3
    _, e2c, d2c = add_micro_safe_divisions(nodes_mp, edges_mp)
    assert d2c["n_divisions_added"] == 0, "candidate already has a parent -> no multi-parent"

    # Test 3: single-frame gap - end 0 @t0 and start 1 @t2, close together,
    # bridged by ONE synthetic node @t1 with two UNIT edges; NO direct t0->t2
    # edge; synthetic ids negative.
    gnodes = pd.DataFrame({"node_id": [0, 1], "t": [0, 2], "z": [0.0, 0.0], "y": [0.0, 8.0], "x": [0.0, 0.0]})
    gedges = pd.DataFrame(columns=["source_id", "target_id"])
    n3, e3, d3, sid3 = close_single_frame_gaps(gnodes, gedges)
    assert d3["n_gap1_closed"] == 1 and len(n3) == 3, "one synthetic node inserted"
    synth = n3[n3["node_id"] < 0]
    assert len(synth) == 1 and int(synth.iloc[0]["t"]) == 1, "synthetic node at t+1 with negative id"
    assert (0, 1) not in set(zip(e3["source_id"], e3["target_id"])), "NO direct t->t+2 edge"
    _assert_edges_unit_t(n3, e3, "gap1"); _assert_topology_ok(n3, e3, "gap1")

    # Test 4: two-frame gap - predecessor p(t0)->e(t1) sets an incoming
    # velocity aligned with the gap; start s @t4; two synthetic nodes @t2,t3
    # with three UNIT edges; no direct multi-frame edge.
    tn = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 1, 4], "z": [0.0] * 3, "y": [0.0, 2.0, 8.0], "x": [0.0] * 3})
    te = pd.DataFrame({"source_id": [0], "target_id": [1]})  # p=0 -> e=1
    n4, e4, d4, sid4 = recover_two_frame_gaps(tn, te)
    assert d4["n_gap2_recovered"] == 1 and int((n4["node_id"] < 0).sum()) == 2, "two synthetic nodes inserted"
    synth_ts = sorted(int(t) for t in n4[n4["node_id"] < 0]["t"])
    assert synth_ts == [2, 3], "synthetic nodes at t+1 and t+2"
    assert (1, 2) not in set(zip(e4["source_id"], e4["target_id"])), "NO direct t1->t4 edge"
    _assert_edges_unit_t(n4, e4, "gap2"); _assert_topology_ok(n4, e4, "gap2")

    # Test 4b: velocity gate rejects a reversed gap. Incoming velocity points
    # +y; the start sits far in -y so cos < velocity_cos_min -> no recovery.
    tn_r = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 1, 4], "z": [0.0] * 3, "y": [0.0, 2.0, -8.0], "x": [0.0] * 3})
    _, _, d4b, _ = recover_two_frame_gaps(tn_r, pd.DataFrame({"source_id": [0], "target_id": [1]}))
    assert d4b["n_gap2_recovered"] == 0, "reversed-velocity gap rejected"

    # Test 5: isolated-node pruning removes ONLY degree-0 nodes; the safety
    # valve skips pruning when too many nodes are isolated.
    pn = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 1, 0], "z": [0.0] * 3, "y": [0.0, 1.0, 5.0], "x": [0.0] * 3})
    pe = pd.DataFrame({"source_id": [0], "target_id": [1]})  # node 2 isolated
    p5n, p5e, d5 = prune_isolated_nodes(pn, pe)
    assert d5["n_pruned"] == 1 and set(p5n["node_id"]) == {0, 1}, "only the degree-0 node removed"
    all_iso = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 0, 0], "z": [0.0] * 3, "y": [0.0, 1.0, 2.0], "x": [0.0] * 3})
    p5n2, _, d5b = prune_isolated_nodes(all_iso, pd.DataFrame(columns=["source_id", "target_id"]))
    assert d5b["n_pruned"] == 0 and d5b["prune_skipped_reason"] and len(p5n2) == 3, "safety valve keeps all when mostly isolated"

    # Test 6: relabel_positive -> positive consecutive ids from start_id;
    # edges remapped; an edge to a removed node is dropped.
    rn = pd.DataFrame({"node_id": [-2, 0, 5], "t": [1, 0, 2], "z": [0.0] * 3, "y": [0.0, 1.0, 2.0], "x": [0.0] * 3})
    re = pd.DataFrame({"source_id": [0, -2], "target_id": [-2, 99]})  # second edge dangling (99 absent)
    r6n, r6e, nxt = relabel_positive(rn, re, start_id=10)
    assert list(r6n["node_id"]) == [10, 11, 12] and nxt == 13, "positive consecutive from start_id"
    assert (r6n["node_id"] >= 0).all(), "no negative ids survive"
    assert len(r6e) == 1 and set(zip(r6e["source_id"], r6e["target_id"])) == {(11, 10)}, "dangling edge dropped, valid one remapped"

    # Test 7: full chain stays metric-valid and topology-valid end to end.
    # Unit-t chain 0->1->3 (only unit base edges); orphan 2 @t1 is a valid
    # division sister for source 0; end 3 @t2 and start 4 @t4 are a single-
    # frame gap. The chain must add only unit edges and keep topology valid.
    cn = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4],
        "t":       [0, 1, 1, 2, 4],
        "z": [0.0] * 5, "y": [0.0, 1.0, 2.0, 2.0, 6.0], "x": [0.0] * 5,
    })
    ce = pd.DataFrame({"source_id": [0, 1], "target_id": [1, 3]})  # 0->1->3 (all unit)
    fn, fe, fdiag = apply_postprocess_chain(cn, ce, M20_VARIANTS["A"]["ops"])
    assert fdiag["safe_divisions"]["n_divisions_added"] == 1, "division admitted in full chain"
    assert fdiag["gap1"]["n_gap1_closed"] == 1, "single-frame gap closed in full chain"
    _assert_edges_unit_t(fn, fe, "full_chain"); _assert_topology_ok(fn, fe, "full_chain")

    # Test 8: end-to-end GEFF roundtrip. Empty ops == M16 baseline counts;
    # the full chain stays valid with no negative node_ids and unit edges.
    ok = _write_synthetic_geff_store(
        scratch / "pred" / "ds1.geff",
        nodes=pd.DataFrame({"raw_id": [0, 1, 2, 3], "t": [0, 1, 2, 2], "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 20.0], "x": [0.0] * 4}),
        edges=pd.DataFrame({"raw_source": [0, 1], "raw_target": [1, 2]}),
        node_solution=[1, 1, 1, 1], edge_solution=[1, 1], edge_prob=[0.9, 0.8], edge_dist=[0.4, 0.4])
    if ok:
        base_df, base_conv = convert_predictions_to_submission_postprocessed(scratch / "pred", ["ds1"], ops=[])
        m16_df, _ = convert_predictions_to_submission(scratch / "pred", ["ds1"])
        assert (base_df["row_type"] == "node").sum() == (m16_df["row_type"] == "node").sum(), "empty ops == M16 node count"
        assert (base_df["row_type"] == "edge").sum() == (m16_df["row_type"] == "edge").sum(), "empty ops == M16 edge count"
        v0, _, _ = validate_reference_submission(base_df, ["ds1"])
        assert v0, "empty-ops submission valid"

        full_df, full_conv = convert_predictions_to_submission_postprocessed(scratch / "pred", ["ds1"], ops=M20_VARIANTS["A"]["ops"])
        vf, _, _ = validate_reference_submission(full_df, ["ds1"])
        assert vf, "full-chain submission valid"
        node_rows = full_df[full_df["row_type"] == "node"]
        assert (node_rows["node_id"] >= 0).all(), "no negative node_id reaches the submission"
        # every submission edge is unit-timepoint.
        t_of = dict(zip(node_rows["node_id"], node_rows["t"]))
        for er in full_df[full_df["row_type"] == "edge"].itertuples():
            assert int(t_of[int(er.target_id)]) - int(t_of[int(er.source_id)]) == 1, "submission edge is unit-timepoint"
    else:
        print("  [test 8] zarr unavailable - skipped GEFF roundtrip (logic covered by tests 1-7)")

    # Test 9: HARD BASELINE GUARD - a matching base passes; the M20 bad-run
    # base (142193/127563) fails; tolerance is 0 (off-by-one fails); a
    # missing count fails closed. This is the fix that makes M20 submit-safe.
    g_ok = check_baseline_guard({"n_nodes_before": EXPECTED_N_NODES_BEFORE, "n_edges_before": EXPECTED_N_EDGES_BEFORE})
    assert g_ok["baseline_guard_passed"] is True and g_ok["submission_recommendation"] == "OK_TO_SUBMIT_IF_GATES_PASS"
    g_bad = check_baseline_guard({"n_nodes_before": 142193, "n_edges_before": 127563})
    assert g_bad["baseline_guard_passed"] is False and g_bad["submission_recommendation"] == "DO_NOT_SUBMIT_BASELINE_MISMATCH"
    assert check_baseline_guard({"n_nodes_before": 131798, "n_edges_before": 118992})["baseline_guard_passed"] is False, "tolerance 0: +1 node must fail"
    assert check_baseline_guard({"n_nodes_before": 131797, "n_edges_before": 118993})["baseline_guard_passed"] is False, "tolerance 0: +1 edge must fail"
    assert check_baseline_guard({})["baseline_guard_passed"] is False, "missing counts must fail closed"
    print("  [test 9] baseline guard: match passes; drift/off-by-one/missing fail (DO_NOT_SUBMIT)")

    print("All milestone20_fullchain_tuned_fixed_runner tests passed (9/9).")

# --------------------------------------------------------------------------- #
# 25. Top-level driver
# --------------------------------------------------------------------------- #
DEFAULT_M20_VARIANT = "A"  # the single tuned + baseline-guarded full_chain candidate


def run_milestone20_variant(
    variant_name: str = DEFAULT_M20_VARIANT, working_dir: str = KAGGLE_WORKING_DIR,
    competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
) -> dict:
    """Runs the self-tests always. On Kaggle (definitive marker
    /kaggle/input present) runs ONE variant through the top-level fallback
    (guaranteeing a valid submission.csv, or re-raising only if even the
    fallback fails). Off Kaggle, stays a clean self-test-only dry run.
    """
    print("=== Self-test: metric-aware post-processing (divisions, gaps, linefit, prune) + reused cores ===")
    run_milestone20_tests()

    if variant_name not in M20_VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}, expected one of {sorted(M20_VARIANTS)}")

    if not is_kaggle_env():
        print(
            f"[dry-run] /kaggle/input absent - self-tests only (variant {variant_name}). The reference "
            "learned-graph pipeline runs only on Kaggle where the support-pack artifact is mounted and a "
            "GPU is available. No submission.csv is written in a dry run."
        )
        return {"status": "dry_run", "variant": variant_name}

    return run_m20_variant_with_fallback(variant_name, working_dir, competition_dir)


if __name__ == "__main__":
    run_milestone20_variant(DEFAULT_M20_VARIANT)
