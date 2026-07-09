"""
Biohub - Cell Tracking During Development
Milestone 27 - PUBLIC NOTEBOOK reproduction + edge-policy-veto pack.

A public high-scoring notebook (biohub-cell-tracking-v4-unet-ilp-reproduction)
runs the SAME 400ep artifact we tested in M24 (artifact_name contains "400ep",
weight_sha256 12f6881e..), but reaches a much better public score by REPLACING
our thin safe_div/gap/linefit post-processing with a richer, evaluator-safe
output pipeline. M24 scored only 0.872/0.873 on 400ep because our postprocess
was too thin. This pack reproduces the public notebook's LOGIC (not its static
submission.csv) and makes it hidden-rerun safe.

Public output pipeline (per dataset, all metric-valid: unit-timepoint edges,
in_degree<=1, out_degree<=2):
  1. motion_relink_edges       - replace the raw learned edges with a per-frame
     Hungarian assignment on motion-predicted distance (pos + velocity_weight *
     predecessor velocity) minus a learned edge_prob bonus; feasible only within
     the relaxed radius. Gives a clean in<=1/out<=1 skeleton.
  2. close_single_frame_gaps   - bridge an END@t and START@t+2 with ONE node at
     t+1; reuse an existing unmatched t+1 node when one sits within reuse_um
     (fewer synthetic nodes), else insert a synthetic midpoint. Capped.
  3. add_safe_divisions        - admit a geometrically safe SECOND child
     (out_degree 1 -> 2) under per-frame/global caps.
  4. apply_edge_policy_veto    - an embedded logistic edge-only policy scores
     every scored edge; the weakest are vetoed as a CONSERVATIVE repair, capped
     at max_veto_frac (1%) of scored edges, never vetoing a division source.
  5. filter_short_track_components - drop connected components shorter than
     min_track_len UNLESS they contain a division (keep_division_components).
  6. linefit_smooth            - topology-preserving coordinate smoothing.
Then a FINAL SAFETY REPAIR enforces in<=1 / out<=2 (keep the two best children
by edge_prob, then repair_policy_score, then shorter distance_um), all edges
t->t+1, no dangling edges, no NaN, consecutive ids.

The 400ep artifact is GUARDED (path/name "400ep" + exact weight_sha256); if it
is absent or the base drifts, the run refuses to submit and a valid hidden-safe
fallback submission is still written.

Four variants (submit order A -> C -> B -> D):
  - A exact_safety     : faithful reproduction + final safety repair. Submit
       first when OK_TO_SUBMIT_EXPERIMENTAL.
  - C minlen5          : min_track_len 5 (does minlen 7 over-prune true short
       tracks?). Submit if valid and n_node_rows <= 126000.
  - B no_edge_veto     : edge-policy veto disabled (does the veto help or hurt?).
       Submit if valid and counts sane.
  - D strict_precision : min_track_len 8 + keep divisions (does an even lower
       node penalty raise the score?). Submit if n_node_rows>=112000 and
       n_edge_rows>=108000.

OK_TO_SUBMIT_EXPERIMENTAL requires: artifact_guard_passed, valid, no fallback,
final_source=public_notebook_logic_reproduced, no NaN, consecutive id, no
dangling edges, all edges unit-timepoint, max_in<=1 / max_out<=2,
110000<=n_node_rows<=130000, 105000<=n_edge_rows<=122000, synthetic gap nodes
<=2600, no dataset fully emptied by repair, all expected dynamic test datasets
present. Otherwise a specific DO_NOT_SUBMIT_* is reported and the fallback is
written. All outputs are generated from mounted Kaggle inputs at runtime; no
static public-test submission is ever used; no Kaggle-API submit; off Kaggle it
is a clean self-test-only dry run. M19-C 0.880 stays final unless M27 beats it.
No prior milestone file is modified.
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
# 21. Reused metric primitives (physical_distance_um, relabel_positive)
# --------------------------------------------------------------------------- #
# All gates are physical micrometres (VOXEL_SIZE_UM / _VOXEL_SCALE come from
# the reused M16 header). Every added edge spans a UNIT timepoint (t -> t+1)
# so the evaluator can credit it; a direct t -> t+2 edge is never produced.

# Safe-division gates (geometric second-child admission).
M27_DIV_PARENT_CHILD_MAX_UM = 4.7    # source -> new (second) child
M27_DIV_SISTER_MAX_UM = 6.85         # existing child <-> new child
M27_DIV_EXISTING_CHILD_MAX_UM = 7.45  # the original single edge must not already be over-stretched
M27_DIV_FRAME_CAP_FRAC = 0.0072      # per-timepoint cap on admitted divisions
M27_DIV_GLOBAL_CAP_FRAC = 0.004      # per-dataset cap on admitted divisions

# Single-frame gap recovery (end@t -> synthetic@t+1 -> start@t+2).
M27_GAP1_MAX_TOTAL_UM = 6.2
M27_GAP1_CAP_FRAC = 0.006
M27_GAP1_CAP_ABS = 300

# Two-frame gap recovery (end@t -> synth@t+1 -> synth@t+2 -> start@t+3).
M27_GAP2_MAX_STEP_UM = 4.4           # per-unit-step budget (total <= 3*step by construction)
M27_GAP2_MAX_TOTAL_UM = 10.2
M27_GAP2_VELOCITY_COS_MIN = -0.25    # incoming velocity vs gap velocity must not strongly reverse
M27_GAP2_VELOCITY_NORMDIFF_UM = 6.0  # |per-step gap velocity - incoming velocity| budget
M27_GAP2_CAP_FRAC = 0.0045
M27_GAP2_CAP_ABS = 180

# Line-fit smoothing (topology-preserving coordinate blend).
M27_LINEFIT_WINDOW = 2
M27_LINEFIT_WEIGHT = 0.72

# Isolated-node pruning (degree-0 predicted nodes = single-frame tracks).
M27_PRUNE_MAX_FRAC = 0.6             # safety valve: if MORE than this fraction is isolated, skip (likely a read problem, not over-prediction)


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
    parent_child_max_um: float = M27_DIV_PARENT_CHILD_MAX_UM, sister_max_um: float = M27_DIV_SISTER_MAX_UM,
    existing_child_max_um: float = M27_DIV_EXISTING_CHILD_MAX_UM,
    frame_cap_frac: float = M27_DIV_FRAME_CAP_FRAC, global_cap_frac: float = M27_DIV_GLOBAL_CAP_FRAC,
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
    max_total_um: float = M27_GAP1_MAX_TOTAL_UM, cap_frac: float = M27_GAP1_CAP_FRAC, cap_abs: int = M27_GAP1_CAP_ABS,
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
    max_step_um: float = M27_GAP2_MAX_STEP_UM, max_total_um: float = M27_GAP2_MAX_TOTAL_UM,
    velocity_cos_min: float = M27_GAP2_VELOCITY_COS_MIN, velocity_normdiff_um: float = M27_GAP2_VELOCITY_NORMDIFF_UM,
    cap_frac: float = M27_GAP2_CAP_FRAC, cap_abs: int = M27_GAP2_CAP_ABS, next_synth_id: int = -1,
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
    window: int = M27_LINEFIT_WINDOW, weight: float = M27_LINEFIT_WEIGHT,
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
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, max_frac: float = M27_PRUNE_MAX_FRAC,
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

# --------------------------------------------------------------------------- #
# 40. M27 public-notebook output pipeline (hidden-safe, evaluator-aligned)
# --------------------------------------------------------------------------- #
# Every produced edge spans a UNIT timepoint (t -> t+1); in_degree<=1 and
# out_degree<=2 are enforced; all gates are physical micrometres via
# physical_distance_um (VOXEL_SIZE_UM from the reused M16 header). A direct
# multi-frame edge is never created. Synthetic ids are negative until the final
# relabel makes every id positive-consecutive.
M27_NODE_SCORE_FIELDS = ["score", "node_score", "node_prob", "prob", "weight", "confidence", "detection_score"]
M27_EDGE_POLICY_COEF = {"bias": 0.55, "edge_prob": 3.2, "edge_dist_um": -0.42, "motion_cos": 0.8}
M27_FINAL_SOURCE = "public_notebook_logic_reproduced"


def read_node_scores_m27(geff_root: Path) -> tuple[dict | None, str | None]:
    """Per-node detection score keyed by raw id (solution-masked when present)."""
    ids = _read_geff_array(geff_root, "nodes/ids")
    if ids is None:
        return None, None
    ids = np.asarray(ids).reshape(-1)
    score = None
    field = None
    for name in M27_NODE_SCORE_FIELDS:
        arr = _read_geff_array(geff_root, f"nodes/props/{name}/values")
        if arr is not None:
            score = np.asarray(arr).reshape(-1).astype(float)
            field = name
            break
    if score is None or len(score) != len(ids):
        return None, None
    sol = _read_geff_array(geff_root, "nodes/props/solution/values")
    if sol is not None:
        mask = np.asarray(sol).reshape(-1).astype(bool)
        ids = ids[mask]
        score = score[mask]
    return {int(i): float(s) for i, s in zip(ids, score)}, field


def build_base_graph_m27(geff_graph: dict, raw_score_map: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M16-identical topology (raw -> consecutive local id; drop dangling /
    backward / duplicate edges; in<=1 keeping best incoming; out<=2 keeping the
    two best) but retains per-node score and per-edge edge_prob/edge_dist so the
    motion-relink and edge-policy steps have their features.
    pred_nodes: node_id,t,z,y,x,score.  pred_edges: source_id,target_id,edge_prob,edge_dist."""
    nodes = geff_graph["nodes"].copy()
    edges = geff_graph["edges"].copy()
    raw_to_local: dict[int, int] = {}
    node_rows = []
    node_t: dict[int, int] = {}
    for local_id, row in enumerate(nodes.itertuples()):
        raw_to_local[int(row.raw_id)] = local_id
        node_t[local_id] = int(row.t)
        score = raw_score_map.get(int(row.raw_id)) if raw_score_map is not None else None
        node_rows.append({"node_id": local_id, "t": int(row.t), "z": float(row.z), "y": float(row.y),
                          "x": float(row.x), "score": score})
    pred_nodes = pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x", "score"])
    kept = []
    for row in edges.itertuples():
        s_raw, t_raw = int(row.raw_source), int(row.raw_target)
        if s_raw not in raw_to_local or t_raw not in raw_to_local:
            continue
        s, t = raw_to_local[s_raw], raw_to_local[t_raw]
        if not (node_t[t] > node_t[s]):
            continue
        kept.append({"source_id": s, "target_id": t, "edge_prob": float(row.edge_prob), "edge_dist": float(row.edge_dist)})
    edges_df = pd.DataFrame(kept, columns=["source_id", "target_id", "edge_prob", "edge_dist"])
    if len(edges_df):
        edges_df = edges_df.drop_duplicates(subset=["source_id", "target_id"], keep="first")
        edges_df = edges_df.assign(_order=np.arange(len(edges_df)))
        edges_df = edges_df.sort_values(["edge_prob", "edge_dist", "_order"], ascending=[False, True, True]).reset_index(drop=True)
        edges_df = edges_df.drop_duplicates(subset=["target_id"], keep="first")
        edges_df = edges_df.groupby("source_id", sort=False, group_keys=False).head(2).reset_index(drop=True)
        edges_df = edges_df[["source_id", "target_id", "edge_prob", "edge_dist"]]
    return pred_nodes, edges_df


def _m27_component_map(node_ids, sources, targets) -> dict:
    """Union-find: node_id -> component root (undirected connectivity)."""
    parent = {int(n): int(n) for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, t in zip(sources, targets):
        s, t = int(s), int(t)
        if s in parent and t in parent:
            rs, rt = find(s), find(t)
            if rs != rt:
                parent[rs] = rt
    return {n: find(n) for n in parent}


def _nodes_by_t(pred_nodes: pd.DataFrame) -> dict:
    by_t: dict[int, list[int]] = {}
    for row in pred_nodes.itertuples():
        by_t.setdefault(int(row.t), []).append(int(row.node_id))
    return by_t


def _assign_min_cost(cost: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one min-cost assignment. Uses scipy Hungarian when available,
    else a deterministic greedy fallback. Returns (row, col) pairs."""
    n_r, n_c = cost.shape
    if n_r == 0 or n_c == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        order = sorted(((float(cost[i, j]), i, j) for i in range(n_r) for j in range(n_c)), key=lambda z: z[0])
        used_r, used_c, pairs = set(), set(), []
        for _, i, j in order:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
        return pairs


def motion_relink_edges(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    tight_um: float, relaxed_um: float, velocity_weight: float, learned_bonus: float,
) -> tuple[pd.DataFrame, dict]:
    """Replace the raw learned edges with a per-frame Hungarian assignment. For
    each node at t a motion-predicted next position pos + velocity_weight *
    (pos - predecessor_pos) is compared to every node at t+1; the assignment
    cost is that physical distance minus learned_bonus * edge_prob when a raw
    learned edge existed. Only pairs within relaxed_um are feasible. The result
    has in_degree<=1 and out_degree<=1 (divisions are added later). Returns a
    fresh edge frame carrying edge_prob/edge_dist for downstream steps.
    """
    diag = {"n_relinked_edges": 0, "n_frames": 0, "n_tight": 0}
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    by_t = _nodes_by_t(pred_nodes)
    pred_of = dict(zip(pred_edges["target_id"], pred_edges["source_id"])) if len(pred_edges) else {}
    learned = {}
    if len(pred_edges):
        for row in pred_edges.itertuples():
            learned[(int(row.source_id), int(row.target_id))] = (float(row.edge_prob), float(row.edge_dist))
    BIG = 1e6
    new_rows = []
    for t in sorted(by_t.keys()):
        srcs = by_t.get(t, [])
        tgts = by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        diag["n_frames"] += 1
        # motion-predicted source positions.
        pred_pos = {}
        for s in srcs:
            p = pos.loc[s].to_numpy(float)
            v = np.zeros(3, dtype=float)
            if s in pred_of and pred_of[s] in pos.index:
                v = p - pos.loc[pred_of[s]].to_numpy(float)
            pred_pos[s] = p + velocity_weight * v
        cost = np.full((len(srcs), len(tgts)), BIG, dtype=float)
        for i, s in enumerate(srcs):
            for j, tg in enumerate(tgts):
                d = physical_distance_um(pred_pos[s], pos.loc[tg].to_numpy(float))
                if d > relaxed_um:
                    continue
                prob = learned.get((s, tg), (0.0, d))[0]
                cost[i, j] = d - learned_bonus * prob
        for i, j in _assign_min_cost(cost):
            if cost[i, j] >= BIG:
                continue
            s, tg = srcs[i], tgts[j]
            d = physical_distance_um(pos.loc[s].to_numpy(float), pos.loc[tg].to_numpy(float))
            prob, _rawd = learned.get((s, tg), (0.0, d))
            if d <= tight_um:
                diag["n_tight"] += 1
            new_rows.append({"source_id": s, "target_id": tg, "edge_prob": float(prob), "edge_dist": float(d)})
    relinked = pd.DataFrame(new_rows, columns=["source_id", "target_id", "edge_prob", "edge_dist"])
    diag["n_relinked_edges"] = int(len(relinked))
    return relinked, diag


def _edges_deg(edges: pd.DataFrame) -> tuple[dict, dict]:
    out_d = edges.groupby("source_id").size().to_dict() if len(edges) else {}
    in_d = edges.groupby("target_id").size().to_dict() if len(edges) else {}
    return out_d, in_d


def close_single_frame_gaps_m27(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    gap_um: float, reuse_existing: bool, reuse_um: float, max_added_frac: float, max_added_abs: int,
    next_synth_id: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    """Bridge an END@t (out_degree 0) and START@t+2 (in_degree 0) within gap_um
    by a node at t+1 plus two unit-timepoint edges. If reuse_existing and an
    unmatched (degree-0) node already sits at t+1 within reuse_um of the
    straight-line midpoint, REUSE it (no synthetic node); otherwise insert a
    synthetic midpoint. Never a direct t->t+2 edge. Closest-first, capped by
    min(max_added_abs, max_added_frac * n_nodes)."""
    diag = {"gap_candidates": 0, "gap_pairs_selected": 0, "gap_inserted_synthetic": 0,
            "gap_reused_existing": 0, "gap_added_edges": 0}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id
    out_d, in_d = _edges_deg(pred_edges)
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    max_t, min_t = int(pred_nodes["t"].max()), int(pred_nodes["t"].min())
    ends = [(int(n), int(t_of[n])) for n in pred_nodes["node_id"] if out_d.get(n, 0) == 0 and t_of[n] < max_t]
    starts_by_t: dict[int, list[int]] = {}
    for n in pred_nodes["node_id"]:
        if in_d.get(n, 0) == 0 and t_of[n] > min_t:
            starts_by_t.setdefault(int(t_of[n]), []).append(int(n))
    # unmatched (degree-0) nodes per timepoint, for reuse.
    free_by_t: dict[int, list[int]] = {}
    for n in pred_nodes["node_id"]:
        if out_d.get(n, 0) == 0 and in_d.get(n, 0) == 0:
            free_by_t.setdefault(int(t_of[n]), []).append(int(n))

    candidates = []
    for end_id, end_t in ends:
        e_pos = pos.loc[end_id].to_numpy(float)
        for start_id in starts_by_t.get(end_t + 2, []):
            total = physical_distance_um(e_pos, pos.loc[start_id].to_numpy(float))
            if total <= gap_um:
                candidates.append({"end": end_id, "start": start_id, "mid_t": end_t + 1, "total": total})
    diag["gap_candidates"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag, next_synth_id
    candidates.sort(key=lambda c: c["total"])
    cap = min(int(max_added_abs), max(1, int(max_added_frac * max(len(pred_nodes), 1))))
    new_nodes, new_edges = [], []
    used_ends, used_starts, used_reuse = set(), set(), set()
    sid = next_synth_id
    for c in candidates:
        if diag["gap_pairs_selected"] >= cap:
            break
        if c["end"] in used_ends or c["start"] in used_starts:
            continue
        e_pos = pos.loc[c["end"]].to_numpy(float)
        s_pos = pos.loc[c["start"]].to_numpy(float)
        mid = (e_pos + s_pos) / 2.0
        mid_id = None
        if reuse_existing:
            best_d = None
            for cand in free_by_t.get(c["mid_t"], []):
                if cand in used_reuse or cand == c["end"] or cand == c["start"]:
                    continue
                d = physical_distance_um(mid, pos.loc[cand].to_numpy(float))
                if d <= reuse_um and (best_d is None or d < best_d):
                    best_d, mid_id = d, cand
        if mid_id is not None:
            used_reuse.add(mid_id)
            diag["gap_reused_existing"] += 1
        else:
            mid_id = sid
            sid -= 1
            new_nodes.append({"node_id": mid_id, "t": c["mid_t"], "z": float(mid[0]), "y": float(mid[1]), "x": float(mid[2])})
            diag["gap_inserted_synthetic"] += 1
        new_edges.append({"source_id": c["end"], "target_id": mid_id})
        new_edges.append({"source_id": mid_id, "target_id": c["start"]})
        used_ends.add(c["end"])
        used_starts.add(c["start"])
        diag["gap_pairs_selected"] += 1

    res_nodes = pd.concat([pred_nodes[["node_id", "t", "z", "y", "x"]],
                           pd.DataFrame(new_nodes, columns=["node_id", "t", "z", "y", "x"])],
                          ignore_index=True) if new_nodes else pred_nodes[["node_id", "t", "z", "y", "x"]].copy()
    res_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])],
                          ignore_index=True) if new_edges else pred_edges.copy()
    diag["gap_added_edges"] = len(new_edges)
    return res_nodes, res_edges, diag, sid


def add_safe_divisions_postlink(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    parent_child_max_um: float, sister_max_um: float, existing_child_max_um: float,
    frame_cap_frac: float, global_cap_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Admit a geometrically safe SECOND outgoing edge for a source that
    currently has exactly one unit-timepoint child, turning it into a division.
    The new child must currently be unmatched (in_degree 0). Closest-first under
    per-frame and per-dataset caps. Metric-valid (unit-timepoint sister)."""
    diag = {"safe_division_candidates": 0, "safe_divisions_added": 0}
    if len(pred_edges) == 0 or len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    t_of = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    out_d, in_d = _edges_deg(pred_edges)
    existing_child_of: dict[int, int] = {}
    for row in pred_edges.itertuples():
        s, t = int(row.source_id), int(row.target_id)
        if out_d.get(s, 0) != 1 or t_of[t] != t_of[s] + 1:
            continue
        if physical_distance_um(pos.loc[s].to_numpy(float), pos.loc[t].to_numpy(float)) > existing_child_max_um:
            continue
        existing_child_of[s] = t
    if not existing_child_of:
        return pred_nodes.copy(), pred_edges.copy(), diag
    orphans_by_t: dict[int, list[int]] = {}
    for n in pred_nodes["node_id"]:
        if in_d.get(n, 0) == 0:
            orphans_by_t.setdefault(int(t_of[n]), []).append(int(n))
    candidates = []
    for s, child in existing_child_of.items():
        child_t = t_of[child]
        s_pos = pos.loc[s].to_numpy(float)
        child_pos = pos.loc[child].to_numpy(float)
        for cand in orphans_by_t.get(child_t, []):
            if cand in (child, s):
                continue
            c_pos = pos.loc[cand].to_numpy(float)
            d_parent = physical_distance_um(s_pos, c_pos)
            if d_parent > parent_child_max_um:
                continue
            if physical_distance_um(child_pos, c_pos) > sister_max_um:
                continue
            candidates.append({"source_id": s, "new_child": cand, "t": child_t, "d_parent": d_parent})
    diag["safe_division_candidates"] = len(candidates)
    if not candidates:
        return pred_nodes.copy(), pred_edges.copy(), diag
    candidates.sort(key=lambda c: c["d_parent"])
    frame_totals: dict[int, int] = {}
    for t in t_of.values():
        frame_totals[int(t)] = frame_totals.get(int(t), 0) + 1
    global_cap = max(1, int(global_cap_frac * max(len(pred_nodes), 1)))
    frame_counts: dict[int, int] = {}
    new_edges = []
    used_children: set[int] = set()
    for c in candidates:
        if len(new_edges) >= global_cap:
            break
        s, cand, ft = c["source_id"], c["new_child"], c["t"]
        if cand in used_children or in_d.get(cand, 0) != 0 or out_d.get(s, 0) != 1:
            continue
        frame_cap = max(1, int(frame_cap_frac * max(frame_totals.get(ft, 0), 1)))
        if frame_counts.get(ft, 0) >= frame_cap:
            continue
        new_edges.append({"source_id": s, "target_id": cand})
        out_d[s] = out_d.get(s, 0) + 1
        in_d[cand] = in_d.get(cand, 0) + 1
        frame_counts[ft] = frame_counts.get(ft, 0) + 1
        used_children.add(cand)
    res_edges = pd.concat([pred_edges, pd.DataFrame(new_edges, columns=["source_id", "target_id"])],
                          ignore_index=True) if new_edges else pred_edges.copy()
    diag["safe_divisions_added"] = len(new_edges)
    return pred_nodes.copy(), res_edges, diag


def edge_policy_score(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame) -> pd.Series:
    """Embedded logistic edge-only policy in [0,1] per edge. Higher = more
    trustworthy. Features: edge_prob (when available), physical edge length in
    um, and motion cosine (incoming velocity vs this edge's direction). No
    training data is shipped; the coefficients are fixed and conservative - the
    score is used ONLY to veto the weakest edges, never to add any."""
    if len(pred_edges) == 0:
        return pd.Series([], dtype=float)
    pos = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    pred_of = dict(zip(pred_edges["target_id"], pred_edges["source_id"]))
    c = M27_EDGE_POLICY_COEF
    scores = []
    for row in pred_edges.itertuples():
        s, t = int(row.source_id), int(row.target_id)
        prob = float(getattr(row, "edge_prob", 0.0)) if hasattr(row, "edge_prob") else 0.0
        if s in pos.index and t in pos.index:
            sp = pos.loc[s].to_numpy(float)
            tp = pos.loc[t].to_numpy(float)
            dist_um = physical_distance_um(sp, tp)
            edge_vec = (tp - sp) * _VOXEL_SCALE
            motion_cos = 0.0
            if s in pred_of and pred_of[s] in pos.index:
                in_vec = (sp - pos.loc[pred_of[s]].to_numpy(float)) * _VOXEL_SCALE
                na, nb = float(np.linalg.norm(in_vec)), float(np.linalg.norm(edge_vec))
                if na > 1e-9 and nb > 1e-9:
                    motion_cos = float(np.dot(in_vec, edge_vec) / (na * nb))
        else:
            dist_um, motion_cos = 0.0, 0.0
        z = c["bias"] + c["edge_prob"] * prob + c["edge_dist_um"] * dist_um + c["motion_cos"] * motion_cos
        scores.append(1.0 / (1.0 + np.exp(-z)))
    return pd.Series(scores, index=pred_edges.index, dtype=float)


def apply_edge_policy_veto(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    threshold: float, max_veto_frac: float, skip_division_sources: bool,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Score every edge; propose vetoing those below `threshold`; actually veto
    only the weakest, capped at max_veto_frac of scored edges, never touching an
    edge whose source is a division (out_degree>=2). Returns the surviving edges
    and the per-surviving-edge policy score (for the final degree tie-break)."""
    diag = {"repair_policy_edge_candidates": 0, "repair_policy_edges_proposed_veto": 0,
            "repair_policy_edges_vetoed": 0, "repair_policy_edges_capped": 0}
    if len(pred_edges) == 0:
        return pred_edges.copy(), pd.Series([], dtype=float), diag
    scores = edge_policy_score(pred_nodes, pred_edges)
    diag["repair_policy_edge_candidates"] = int(len(pred_edges))
    out_d, _ = _edges_deg(pred_edges)
    proposed = []
    for pos_i, row in enumerate(pred_edges.itertuples()):
        s = int(row.source_id)
        sc = float(scores.iloc[pos_i])
        if sc < threshold:
            if skip_division_sources and out_d.get(s, 0) >= 2:
                continue
            proposed.append((sc, pos_i))
    diag["repair_policy_edges_proposed_veto"] = len(proposed)
    cap = int(max_veto_frac * len(pred_edges))
    proposed.sort(key=lambda z: z[0])  # weakest first
    veto_positions = {p for _, p in proposed[:cap]} if cap > 0 else set()
    diag["repair_policy_edges_vetoed"] = len(veto_positions)
    diag["repair_policy_edges_capped"] = max(0, len(proposed) - len(veto_positions))
    keep_mask = np.array([i not in veto_positions for i in range(len(pred_edges))], dtype=bool)
    survivors = pred_edges.iloc[keep_mask].reset_index(drop=True)
    surv_scores = scores.iloc[keep_mask].reset_index(drop=True)
    return survivors, surv_scores, diag


def filter_short_track_components(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    min_track_len: int, keep_division_components: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove connected components with fewer than min_track_len nodes, UNLESS
    (keep_division_components) the component contains a division node
    (out_degree>=2). Removing whole short components lowers the predicted-node
    penalty; kept components are unchanged."""
    diag = {"short_track_components_removed": 0, "short_track_nodes_removed": 0, "short_track_edges_removed": 0}
    if len(pred_nodes) == 0:
        return pred_nodes.copy(), pred_edges.copy(), diag
    comp = _m27_component_map(list(pred_nodes["node_id"]),
                              pred_edges["source_id"].tolist() if len(pred_edges) else [],
                              pred_edges["target_id"].tolist() if len(pred_edges) else [])
    from collections import defaultdict
    members = defaultdict(list)
    for n, r in comp.items():
        members[r].append(n)
    out_d, _ = _edges_deg(pred_edges)
    division_roots = set()
    if keep_division_components:
        for n in pred_nodes["node_id"]:
            if out_d.get(int(n), 0) >= 2:
                division_roots.add(comp[int(n)])
    remove_nodes: set[int] = set()
    for root, ns in members.items():
        if len(ns) >= min_track_len:
            continue
        if root in division_roots:
            continue
        remove_nodes.update(int(x) for x in ns)
        diag["short_track_components_removed"] += 1
    if not remove_nodes:
        return pred_nodes.copy(), pred_edges.copy(), diag
    res_nodes = pred_nodes[~pred_nodes["node_id"].isin(remove_nodes)].reset_index(drop=True)
    if len(pred_edges):
        emask = ~(pred_edges["source_id"].isin(remove_nodes) | pred_edges["target_id"].isin(remove_nodes))
        res_edges = pred_edges[emask].reset_index(drop=True)
        diag["short_track_edges_removed"] = int((~emask).sum())
    else:
        res_edges = pred_edges.copy()
    diag["short_track_nodes_removed"] = len(remove_nodes)
    return res_nodes, res_edges, diag


def linefit_smooth_output_graph(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, window: int, weight: float,
) -> tuple[pd.DataFrame, dict]:
    """Topology-preserving coordinate smoothing: only linear-interior nodes
    (exactly one predecessor at t-1 and one successor at t+1) are blended toward
    their neighbours; divisions, branch points and endpoints are untouched."""
    diag = {"linefit_smoothed_nodes": 0}
    if len(pred_nodes) == 0 or len(pred_edges) == 0:
        return pred_nodes.copy(), diag
    succ = dict(zip(pred_edges["source_id"], pred_edges["target_id"]))
    pred = dict(zip(pred_edges["target_id"], pred_edges["source_id"]))
    out_d, in_d = _edges_deg(pred_edges)
    lut = pred_nodes.set_index("node_id")
    t_lut = lut["t"].to_dict()

    def linear_interior(nid) -> bool:
        if out_d.get(nid, 0) != 1 or in_d.get(nid, 0) != 1:
            return False
        s_id, p_id = succ.get(nid), pred.get(nid)
        if s_id is None or p_id is None:
            return False
        return t_lut.get(s_id) == t_lut[nid] + 1 and t_lut.get(p_id) == t_lut[nid] - 1

    def collect(nid, direction, steps):
        out, cur = [], nid
        for _ in range(steps):
            nxt = succ.get(cur) if direction == "succ" else pred.get(cur)
            if nxt is None:
                break
            out.append(nxt)
            if not linear_interior(nxt):
                break
            cur = nxt
        return out

    smoothed = {}
    for nid in pred_nodes["node_id"]:
        if not linear_interior(nid):
            continue
        neigh = collect(nid, "succ", window) + collect(nid, "pred", window)
        if not neigh:
            continue
        neigh_avg = lut.loc[neigh, ["z", "y", "x"]].to_numpy(float).mean(axis=0)
        original = lut.loc[nid, ["z", "y", "x"]].to_numpy(float)
        smoothed[nid] = weight * original + (1.0 - weight) * neigh_avg
    res = pred_nodes.copy()
    if smoothed:
        idx = res.set_index("node_id")
        for nid, p in smoothed.items():
            idx.loc[nid, ["z", "y", "x"]] = p
        res = idx.reset_index()[["node_id", "t", "z", "y", "x"]]
    diag["linefit_smoothed_nodes"] = len(smoothed)
    return res, diag


def final_safety_repair(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, policy_scores: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Enforce the hard submission invariants that the evaluator (and our
    guards) require: every edge unit-timepoint (t->t+1), no dangling edge, in<=1
    (keep the single best incoming), out<=2 (keep the two best children ranked
    by edge_prob, then repair_policy_score, then shorter distance_um), no NaN
    coordinate. Ids are made positive-consecutive by the caller's relabel."""
    diag = {"dropped_non_unit": 0, "dropped_dangling": 0, "dropped_over_in": 0, "dropped_over_out": 0, "nan_nodes_dropped": 0}
    nodes = pred_nodes.copy()
    # drop NaN-coordinate nodes.
    good = ~nodes[["z", "y", "x"]].isna().any(axis=1)
    diag["nan_nodes_dropped"] = int((~good).sum())
    nodes = nodes[good].reset_index(drop=True)
    valid_ids = set(nodes["node_id"])
    t_of = dict(zip(nodes["node_id"], nodes["t"]))
    pos = nodes.set_index("node_id")[["z", "y", "x"]]
    if len(pred_edges) == 0:
        return nodes, pd.DataFrame(columns=["source_id", "target_id"]), diag
    e = pred_edges.copy()
    n0 = len(e)
    # dangling (endpoint missing).
    e = e[e["source_id"].isin(valid_ids) & e["target_id"].isin(valid_ids)].reset_index(drop=True)
    diag["dropped_dangling"] = n0 - len(e)
    # unit-timepoint only.
    if len(e):
        um = e.apply(lambda r: t_of.get(int(r["target_id"])) == t_of.get(int(r["source_id"])) + 1, axis=1)
        diag["dropped_non_unit"] = int((~um).sum())
        e = e[um].reset_index(drop=True)
    if not len(e):
        return nodes, e[["source_id", "target_id"]] if "source_id" in e else pd.DataFrame(columns=["source_id", "target_id"]), diag
    # tie-break columns.
    if "edge_prob" not in e:
        e["edge_prob"] = 0.0
    if policy_scores is not None and len(policy_scores) == len(pred_edges):
        # align by original position is unsafe after filtering; recompute score.
        e["policy_score"] = edge_policy_score(nodes, e).values
    else:
        e["policy_score"] = 0.0
    if "edge_dist" not in e:
        e["edge_dist"] = [physical_distance_um(pos.loc[int(r.source_id)].to_numpy(float),
                                               pos.loc[int(r.target_id)].to_numpy(float)) for r in e.itertuples()]
    e = e.assign(_o=np.arange(len(e)))
    e = e.sort_values(["edge_prob", "policy_score", "edge_dist", "_o"], ascending=[False, False, True, True]).reset_index(drop=True)
    before = len(e)
    e = e.drop_duplicates(subset=["target_id"], keep="first")   # in_degree <= 1
    diag["dropped_over_in"] = before - len(e)
    before = len(e)
    e = e.groupby("source_id", sort=False, group_keys=False).head(2).reset_index(drop=True)  # out_degree <= 2
    diag["dropped_over_out"] = before - len(e)
    return nodes, e[["source_id", "target_id"]].reset_index(drop=True), diag


def apply_public_pipeline(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the public notebook's output pipeline in order (motion relink -> gap
    close -> safe divisions -> edge-policy veto -> short-track filter -> linefit)
    then the final safety repair. `cfg` toggles each stage and carries its
    gates. Returns (nodes[node_id,t,z,y,x], edges[source_id,target_id], diag)."""
    diag: dict = {"stages": []}
    nodes = pred_nodes.copy()
    edges = pred_edges.copy()
    policy_scores = None

    if cfg.get("motion_relink", True):
        edges, d = motion_relink_edges(nodes, edges, cfg["motion_relink_tight_um"], cfg["motion_relink_relaxed_um"],
                                       cfg["motion_relink_velocity_weight"], cfg["motion_relink_learned_bonus"])
        diag["motion_relink"] = d
        diag["stages"].append("motion_relink")
    nodes = nodes[["node_id", "t", "z", "y", "x"]].copy()

    if cfg.get("gap_close", True):
        nodes, edges, d, _sid = close_single_frame_gaps_m27(
            nodes, edges, cfg["gap_close_um"], cfg["gap_close_reuse_existing"], cfg["gap_close_reuse_um"],
            cfg["gap_close_max_added_frac"], cfg["gap_close_max_added_abs"], next_synth_id=-1)
        diag["gap_close"] = d
        diag["stages"].append("gap_close")

    if cfg.get("safe_divisions", True):
        nodes, edges, d = add_safe_divisions_postlink(
            nodes, edges, cfg["safe_div_max_um"], cfg["safe_div_sister_max_um"], cfg["safe_div_existing_child_max_um"],
            cfg["safe_div_frame_frac_cap"], cfg["safe_div_global_frac_cap"])
        diag["safe_divisions"] = d
        diag["stages"].append("safe_divisions")

    if cfg.get("edge_policy_veto", True):
        edges, policy_scores, d = apply_edge_policy_veto(
            nodes, edges, cfg["repair_policy_edge_veto_threshold"], cfg["repair_policy_max_veto_frac"],
            cfg["repair_policy_skip_division_sources"])
        diag["edge_policy_veto"] = d
        diag["stages"].append("edge_policy_veto")
    else:
        diag["edge_policy_veto"] = {"repair_policy_edge_candidates": int(len(edges)),
                                    "repair_policy_edges_proposed_veto": 0, "repair_policy_edges_vetoed": 0,
                                    "repair_policy_edges_capped": 0, "disabled": True}

    if cfg.get("filter_short_tracks", True):
        nodes, edges, d = filter_short_track_components(nodes, edges, cfg["min_track_len"], cfg["keep_division_components"])
        diag["filter_short_tracks"] = d
        diag["stages"].append("filter_short_tracks")

    if cfg.get("linefit_smooth", True):
        nodes, d = linefit_smooth_output_graph(nodes, edges, cfg["linefit_window"], cfg["linefit_weight"])
        diag["linefit_smooth"] = d
        diag["stages"].append("linefit_smooth")

    nodes, edges, d = final_safety_repair(nodes, edges, policy_scores)
    diag["final_safety_repair"] = d
    diag["stages"].append("final_safety_repair")
    return nodes, edges, diag


def convert_public_pipeline_to_submission(
    predictions_dir: Path, expected_datasets: Sequence[str], cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Per dataset: read GEFF -> build_base_graph_m27 -> apply_public_pipeline
    -> relabel to globally-consecutive positive ids -> emit node then edge rows.
    Aggregates the public-notebook diagnostic fields the report needs."""
    stores = find_prediction_geff_stores(predictions_dir)
    all_rows = []
    per_dataset = []
    next_id = next_node_id = 0
    agg = {k: 0 for k in [
        "raw_nodes", "raw_edges", "final_nodes", "final_edges",
        "gap_candidates", "gap_pairs_selected", "gap_inserted_synthetic", "gap_added_edges",
        "safe_division_candidates", "safe_divisions_added",
        "repair_policy_edge_candidates", "repair_policy_edges_proposed_veto",
        "repair_policy_edges_vetoed", "repair_policy_edges_capped",
        "short_track_components_removed", "short_track_nodes_removed", "short_track_edges_removed",
        "linefit_smoothed_nodes"]}
    empties = []
    for dataset_name, geff_root in stores:
        geff_graph = read_geff_graph(geff_root)
        raw_score_map, _ = read_node_scores_m27(geff_root)
        base_nodes, base_edges = build_base_graph_m27(geff_graph, raw_score_map)
        raw_n, raw_e = len(base_nodes), len(base_edges)
        pp_nodes, pp_edges, pp_diag = apply_public_pipeline(base_nodes, base_edges, cfg)
        pp_nodes, pp_edges, next_node_id = relabel_positive(pp_nodes, pp_edges, next_node_id)
        if len(pp_nodes) == 0:
            empties.append(dataset_name)

        for row in pp_nodes.itertuples():
            all_rows.append({"id": next_id, "dataset": dataset_name, "row_type": "node", "node_id": int(row.node_id),
                             "t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x), "source_id": -1, "target_id": -1})
            next_id += 1
        for row in (pp_edges.itertuples() if len(pp_edges) else []):
            all_rows.append({"id": next_id, "dataset": dataset_name, "row_type": "edge", "node_id": -1,
                             "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(row.source_id), "target_id": int(row.target_id)})
            next_id += 1

        agg["raw_nodes"] += raw_n
        agg["raw_edges"] += raw_e
        agg["final_nodes"] += len(pp_nodes)
        agg["final_edges"] += len(pp_edges)
        for stage, keys in {
            "gap_close": ["gap_candidates", "gap_pairs_selected", "gap_inserted_synthetic", "gap_added_edges"],
            "safe_divisions": ["safe_division_candidates", "safe_divisions_added"],
            "edge_policy_veto": ["repair_policy_edge_candidates", "repair_policy_edges_proposed_veto",
                                 "repair_policy_edges_vetoed", "repair_policy_edges_capped"],
            "filter_short_tracks": ["short_track_components_removed", "short_track_nodes_removed", "short_track_edges_removed"],
            "linefit_smooth": ["linefit_smoothed_nodes"],
        }.items():
            sd = pp_diag.get(stage, {})
            for k in keys:
                agg[k] += int(sd.get(k, 0))
        per_dataset.append({"dataset": dataset_name, "raw_nodes": raw_n, "raw_edges": raw_e,
                            "final_nodes": len(pp_nodes), "final_edges": len(pp_edges), "diag": pp_diag})
        print(f"  [m27] {dataset_name}: raw {raw_n}/{raw_e} -> final {len(pp_nodes)}/{len(pp_edges)} "
              f"(gap_synth={pp_diag.get('gap_close', {}).get('gap_inserted_synthetic', 0)}, "
              f"div+={pp_diag.get('safe_divisions', {}).get('safe_divisions_added', 0)}, "
              f"veto={pp_diag.get('edge_policy_veto', {}).get('repair_policy_edges_vetoed', 0)}, "
              f"short_rm={pp_diag.get('filter_short_tracks', {}).get('short_track_nodes_removed', 0)})")

    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    conversion = {
        "predictions_dir": str(predictions_dir), "n_stores_found": len(stores),
        "found_datasets": [n for n, _ in stores], "expected_datasets": list(expected_datasets),
        "emptied_datasets": empties, **agg, "per_dataset": per_dataset,
    }
    return submission_df, conversion

# --------------------------------------------------------------------------- #
# 43. M27 variant registry + 400ep artifact guard + sanity thresholds
# --------------------------------------------------------------------------- #
M27_400EP_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
M27_400EP_NAME_SUBSTR = "400ep"
M27_PREDICT_REL = "repo/scripts/predict_unet_transformer.py"
M27_WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"

# Public-test base reference for the 400ep artifact (M24 diagnostic).
M27_400EP_BASE = (127790, 115694)

# Submission sanity window (safety gate).
M27_MIN_NODE_ROWS, M27_MAX_NODE_ROWS = 110000, 130000
M27_MIN_EDGE_ROWS, M27_MAX_EDGE_ROWS = 105000, 122000
M27_MAX_SYNTHETIC_GAP = 2600

# The exact public-notebook config (variant A). Every other variant is A with a
# single documented override.
M27_BASE_CONFIG = {
    "motion_relink": True, "motion_relink_tight_um": 6.0, "motion_relink_relaxed_um": 10.0,
    "motion_relink_velocity_weight": 0.5, "motion_relink_learned_bonus": 0.75,
    "gap_close": True, "gap_close_um": 6.0, "gap_close_reuse_existing": True, "gap_close_reuse_um": 3.2,
    "gap_close_max_added_frac": 0.05, "gap_close_max_added_abs": 2000,
    "gap2_recovery": False,
    "safe_divisions": True, "safe_div_max_um": 4.7, "safe_div_sister_max_um": 7.2,
    "safe_div_existing_child_max_um": 7.8, "safe_div_frame_frac_cap": 0.008, "safe_div_global_frac_cap": 0.004,
    "edge_policy_veto": True, "repair_policy_edge_veto_threshold": 0.35, "repair_policy_max_veto_frac": 0.01,
    "repair_policy_skip_division_sources": True,
    "filter_short_tracks": True, "min_track_len": 7, "keep_division_components": True,
    "linefit_smooth": True, "linefit_weight": 0.8, "linefit_window": 2,
}


def _m27_cfg(**overrides) -> dict:
    cfg = dict(M27_BASE_CONFIG)
    cfg.update(overrides)
    return cfg


M27_VARIANTS = {
    # A: faithful reproduction of the public notebook + final safety repair.
    "A": {"label": "public_repro_exact_safety", "config": _m27_cfg(),
          "submit_rule": "ok_to_submit_experimental"},
    # B: same as A but the edge-policy veto is disabled (isolate its effect).
    "B": {"label": "public_repro_no_edge_veto", "config": _m27_cfg(edge_policy_veto=False),
          "submit_rule": "valid_and_counts_sane"},
    # C: same as A but min_track_len 5 (does minlen 7 over-prune true short tracks?).
    "C": {"label": "public_repro_minlen5", "config": _m27_cfg(min_track_len=5),
          "submit_rule": "valid_and_node_rows_le_126000"},
    # D: same as A but stricter short-track filtering (min_track_len 8, keep divisions).
    "D": {"label": "public_repro_strict_precision", "config": _m27_cfg(min_track_len=8, keep_division_components=True),
          "submit_rule": "node_rows_ge_112000_and_edge_rows_ge_108000"},
}


def _sha256_file_m27(path) -> str | None:
    import hashlib
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest_name_m27(root):
    mp = Path(root) / "ARTIFACT_MANIFEST.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    return m.get("artifact_name") or m.get("name") or m.get("artifact") or m.get("dataset")


def enumerate_artifact_roots_m27(input_root: str = "/kaggle/input") -> list[Path]:
    """Hidden-safe glob of candidate roots 1-3 levels under /kaggle/input
    (datasets AND notebooks), de-duplicated, no hardcoded dataset names."""
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
            if str(p) not in seen:
                seen.add(str(p))
                roots.append(p)
    return roots


def inspect_400ep_candidate(root: Path) -> dict:
    """artifact_name (path + manifest) + weight_sha256 for the 400ep guard."""
    weights = root / M27_WEIGHTS_REL
    has_predict = (root / M27_PREDICT_REL).exists()
    has_weights = weights.exists()
    sha = _sha256_file_m27(weights) if has_weights else None
    manifest_name = _read_manifest_name_m27(root)
    name_blob = " ".join(str(x).lower() for x in [manifest_name, root.name, str(root)])
    name_has_400ep = M27_400EP_NAME_SUBSTR in name_blob
    sha_matches = (sha == M27_400EP_SHA256)
    return {
        "candidate_path": str(root), "artifact_name": manifest_name,
        "weight_sha256": sha, "has_predict_script": has_predict, "has_weights": has_weights,
        "name_has_400ep": name_has_400ep, "sha_matches_400ep": sha_matches,
        "is_400ep_artifact": bool(has_predict and has_weights and name_has_400ep and sha_matches),
    }


def select_400ep_artifact(input_root: str = "/kaggle/input") -> dict:
    """Find the guarded 400ep artifact: name contains "400ep" AND weight_sha256
    matches exactly. Returns the selection with artifact_guard_passed and the
    full candidate list for the artifact_selection report."""
    candidates = [inspect_400ep_candidate(r) for r in enumerate_artifact_roots_m27(input_root)]
    viable = [c for c in candidates if c["is_400ep_artifact"]]
    chosen = viable[0] if viable else None
    # Fallback diagnostics: name-only or sha-only near-matches (guard still fails).
    sha_only = [c for c in candidates if c["sha_matches_400ep"] and not c["name_has_400ep"]]
    name_only = [c for c in candidates if c["name_has_400ep"] and not c["sha_matches_400ep"]]
    return {
        "artifact_guard_passed": chosen is not None,
        "chosen": chosen,
        "artifact_name": chosen["artifact_name"] if chosen else (name_only[0]["artifact_name"] if name_only else None),
        "weight_sha256": chosen["weight_sha256"] if chosen else (sha_only[0]["weight_sha256"] if sha_only else None),
        "expected_sha256": M27_400EP_SHA256, "expected_name_substr": M27_400EP_NAME_SUBSTR,
        "n_candidates": len(candidates), "n_viable_400ep": len(viable),
        "sha_only_matches": [c["candidate_path"] for c in sha_only],
        "name_only_matches": [c["candidate_path"] for c in name_only],
        "candidates": candidates,
    }


def build_m27_command(
    variant_name: str, introspection: dict,
    data_dir: str = f"{KAGGLE_COMPETITION_INPUT_DIR}/test", splits_filename: str = SPLITS_FILENAME,
) -> tuple[list[str], dict]:
    """The EXACT reference predict command (identical to M16/M24): det 0.99,
    ilp-edge -1.0, appearance/disappearance 0.1, division 1.0, --use-ilp."""
    params = dict(BASE_PREDICT_PARAMS)
    available_splits = list(introspection.get("weight_splits", [])) or [0]
    split = select_weight_split(0, available_splits)
    cmd = build_base_command(params, split, splits_filename, data_dir)
    vdef = M27_VARIANTS[variant_name]
    notes = {"variant": variant_name, "label": vdef["label"], "det_threshold": params["det_threshold"],
             "split_used": split, "available_splits": available_splits,
             "min_track_len": vdef["config"]["min_track_len"],
             "edge_policy_veto": vdef["config"]["edge_policy_veto"],
             "predict_command_identical_to_m16": True}
    return cmd, notes

# --------------------------------------------------------------------------- #
# 44. M27 orchestration (guarded 400ep predict -> public pipeline -> guards) + fallback
# --------------------------------------------------------------------------- #
def _m27_submission_invariants(df: pd.DataFrame, expected_datasets: Sequence[str]) -> dict:
    """Hard invariants the report/guards need, computed directly from the
    emitted submission rows (independent of the pipeline's own bookkeeping)."""
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    inv = {"n_node_rows": int(len(nodes)), "n_edge_rows": int(len(edges)),
           "max_in_degree": 0, "max_out_degree": 0, "direct_multiframe_edges": 0,
           "dangling_edges": 0, "id_consecutive": True, "no_nan": True,
           "datasets_present": sorted(df["dataset"].unique().tolist()),
           "missing_datasets": [], "emptied_datasets": []}
    # id consecutive 0..N-1.
    ids = df["id"].to_numpy()
    inv["id_consecutive"] = bool(len(ids) == 0 or (ids.min() == 0 and ids.max() == len(ids) - 1 and len(set(ids.tolist())) == len(ids)))
    # no NaN in node coordinates.
    if len(nodes):
        inv["no_nan"] = bool(not nodes[["t", "z", "y", "x", "node_id"]].isna().any().any())
    # degrees + timepoint validity + dangling, per dataset (node_ids are per-dataset local).
    for ds, g in df.groupby("dataset"):
        gn = g[g["row_type"] == "node"]
        ge = g[g["row_type"] == "edge"]
        if len(gn) == 0:
            inv["emptied_datasets"].append(ds)
        t_of = dict(zip(gn["node_id"].astype(int), gn["t"].astype(int)))
        if len(ge):
            ind = ge.groupby("target_id").size()
            outd = ge.groupby("source_id").size()
            inv["max_in_degree"] = max(inv["max_in_degree"], int(ind.max()))
            inv["max_out_degree"] = max(inv["max_out_degree"], int(outd.max()))
            for r in ge.itertuples():
                s, t = int(r.source_id), int(r.target_id)
                if s not in t_of or t not in t_of:
                    inv["dangling_edges"] += 1
                elif t_of[t] - t_of[s] != 1:
                    inv["direct_multiframe_edges"] += 1
    inv["missing_datasets"] = [d for d in expected_datasets if d not in set(inv["datasets_present"])]
    return inv


def _m27_recommendation(variant_name: str, guard_passed: bool, valid: bool, fallback_used: bool,
                        inv: dict, synthetic: int) -> str:
    """Universal safety gate then the per-variant extra rule. Returns the single
    recommendation string."""
    if not guard_passed:
        return "DO_NOT_SUBMIT_WRONG_ARTIFACT"
    if not valid or fallback_used or not inv["no_nan"] or not inv["id_consecutive"]:
        return "DO_NOT_SUBMIT_VALIDATION_FAILED"
    if inv["dangling_edges"] > 0 or inv["direct_multiframe_edges"] > 0:
        return "DO_NOT_SUBMIT_VALIDATION_FAILED"
    if inv["max_in_degree"] > 1 or inv["max_out_degree"] > 2:
        return "DO_NOT_SUBMIT_VALIDATION_FAILED"
    if inv["missing_datasets"] or inv["emptied_datasets"]:
        return "DO_NOT_SUBMIT_EMPTY_OR_MISSING_DATASET"
    if not (M27_MIN_NODE_ROWS <= inv["n_node_rows"] <= M27_MAX_NODE_ROWS):
        return "DO_NOT_SUBMIT_UNSANE_NODE_COUNT"
    if not (M27_MIN_EDGE_ROWS <= inv["n_edge_rows"] <= M27_MAX_EDGE_ROWS):
        return "DO_NOT_SUBMIT_UNSANE_EDGE_COUNT"
    if synthetic > M27_MAX_SYNTHETIC_GAP:
        return "DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION"
    # per-variant extra rule.
    if variant_name == "C" and inv["n_node_rows"] > 126000:
        return "DO_NOT_SUBMIT_MINLEN5_NODES_TOO_HIGH"
    if variant_name == "D" and (inv["n_node_rows"] < 112000 or inv["n_edge_rows"] < 108000):
        return "DO_NOT_SUBMIT_STRICT_PRECISION_UNDERCUT"
    return "OK_TO_SUBMIT_EXPERIMENTAL"


def _write_m27_run_stats_csv(out_dir: Path, conversion: dict) -> None:
    rows = []
    for d in conversion["per_dataset"]:
        gc = d["diag"].get("gap_close", {})
        sd = d["diag"].get("safe_divisions", {})
        ev = d["diag"].get("edge_policy_veto", {})
        ft = d["diag"].get("filter_short_tracks", {})
        lf = d["diag"].get("linefit_smooth", {})
        rows.append({
            "dataset": d["dataset"], "raw_nodes": d["raw_nodes"], "raw_edges": d["raw_edges"],
            "final_nodes": d["final_nodes"], "final_edges": d["final_edges"],
            "gap_inserted_synthetic": gc.get("gap_inserted_synthetic", 0), "gap_reused_existing": gc.get("gap_reused_existing", 0),
            "safe_divisions_added": sd.get("safe_divisions_added", 0),
            "repair_policy_edges_vetoed": ev.get("repair_policy_edges_vetoed", 0),
            "short_track_nodes_removed": ft.get("short_track_nodes_removed", 0),
            "linefit_smoothed_nodes": lf.get("linefit_smoothed_nodes", 0),
        })
    rows.append({"dataset": "TOTAL", "raw_nodes": conversion["raw_nodes"], "raw_edges": conversion["raw_edges"],
                 "final_nodes": conversion["final_nodes"], "final_edges": conversion["final_edges"],
                 "gap_inserted_synthetic": conversion["gap_inserted_synthetic"], "gap_reused_existing": "",
                 "safe_divisions_added": conversion["safe_divisions_added"],
                 "repair_policy_edges_vetoed": conversion["repair_policy_edges_vetoed"],
                 "short_track_nodes_removed": conversion["short_track_nodes_removed"],
                 "linefit_smoothed_nodes": conversion["linefit_smoothed_nodes"]})
    pd.DataFrame(rows).to_csv(out_dir / "milestone27_run_stats.csv", index=False)


def run_m27_variant_pipeline(
    variant_name: str, working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
) -> dict:
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = Path(competition_dir)
    test_dir = comp / "test"
    sample_submission_path = comp / "sample_submission.csv"
    vdef = M27_VARIANTS[variant_name]
    cfg = vdef["config"]
    log(f"=== Milestone 27 variant {variant_name} ({vdef['label']}) min_track_len={cfg['min_track_len']} "
        f"edge_policy_veto={cfg['edge_policy_veto']} ===")

    # 400ep artifact guard.
    selection = select_400ep_artifact()
    (out_dir / "milestone27_artifact_selection.json").write_text(json.dumps(selection, indent=2, default=str))
    guard_passed = selection["artifact_guard_passed"]
    log(f"400ep artifact guard: passed={guard_passed} "
        f"(name_substr '{M27_400EP_NAME_SUBSTR}', sha {M27_400EP_SHA256[:12]}..); "
        f"candidates={selection['n_candidates']} viable={selection['n_viable_400ep']}")

    expected_datasets = discover_test_datasets(test_dir, sample_submission_path)["chosen"]

    if not guard_passed:
        report_obj = _m27_base_report(variant_name, vdef, selection, guard_passed=False, inv=None,
                                      conversion=None, valid=False, fallback_used=False, final_source=None,
                                      recommendation="DO_NOT_SUBMIT_WRONG_ARTIFACT",
                                      instruction=("The guarded 400ep artifact (name contains '400ep' AND weight_sha256 "
                                                   f"{M27_400EP_SHA256[:12]}..) was not found. Attach the "
                                                   "biohub-tracking-support-pack-400ep-snapshot-v1 dataset and re-run. Keep M19-C 0.880."))
        (out_dir / "milestone27_reference_submission_report.json").write_text(json.dumps(report_obj, indent=2, default=str))
        raise RuntimeError("400ep artifact guard failed (wrong/absent artifact)")

    # Materialize + deps + splits + predict (exact reference command).
    root = Path(selection["chosen"]["candidate_path"])
    materialize = materialize_repo(root, out_dir)
    write_sitecustomize()
    env = build_subprocess_env()
    repo_dst = Path(materialize["repo_dst"])
    dep_report = install_dependencies(root / "wheels", env)
    if not dep_report.get("all_ok"):
        raise RuntimeError("wheel install failed")
    _ = verify_imports(env)
    shipped_template = repo_dst / SPLITS_FILENAME
    prepare_splits(test_dir=test_dir, sample_submission_path=sample_submission_path,
                   template_path=shipped_template if shipped_template.exists() else None,
                   out_splits_path=repo_dst / SPLITS_FILENAME, diagnostic_path=out_dir / "m27_splits.json")
    cmd, cmd_notes = build_m27_command(variant_name, {"weight_splits": list_weight_splits(root)})
    log("predict command: " + " ".join(cmd))
    exec_result = run_prediction(cmd, cwd=repo_dst, env=env)
    if not exec_result.get("ok"):
        raise RuntimeError("predict returned nonzero")
    predictions_dir = repo_dst / "predictions" / "unknown" / "unet_transformer" / "split_0"

    # Public pipeline conversion.
    submission_df, conversion = convert_public_pipeline_to_submission(predictions_dir, expected_datasets, cfg)
    (out_dir / "milestone27_ensemble_conversion.json").write_text(json.dumps(conversion, indent=2, default=str))

    valid, vreport, vstats = validate_reference_submission(submission_df, expected_datasets)
    log(vreport)
    inv = _m27_submission_invariants(submission_df, expected_datasets)
    synthetic = int(conversion["gap_inserted_synthetic"])

    submission_path = out_dir / "submission.csv"
    if valid:
        submission_df.to_csv(submission_path, index=False)
        submission_df.to_csv(out_dir / f"submission_m27_variant_{variant_name}.csv", index=False)
        write_reference_diagnostics(submission_df, valid, vstats, out_dir)
    _write_m27_run_stats_csv(out_dir, conversion)

    recommendation = _m27_recommendation(variant_name, guard_passed, valid, False, inv, synthetic)
    report_obj = _m27_base_report(variant_name, vdef, selection, guard_passed=True, inv=inv, conversion=conversion,
                                  valid=valid, fallback_used=False, final_source=M27_FINAL_SOURCE,
                                  recommendation=recommendation,
                                  instruction=("submission.csv built (M27 public-notebook logic reproduced, experimental). "
                                               + ("Review the on-Kaggle gates then submit." if recommendation == "OK_TO_SUBMIT_EXPERIMENTAL"
                                                  else f"{recommendation} - keep M19-C 0.880.")))
    (out_dir / "milestone27_reference_submission_report.json").write_text(json.dumps(report_obj, indent=2, default=str))
    (out_dir / "milestone27_failure_fallback_report.json").write_text(json.dumps(
        {"variant": variant_name, "fallback": {"used": False, "reason": "pipeline succeeded"},
         "final_source": M27_FINAL_SOURCE}, indent=2))
    if not valid:
        raise RuntimeError("M27 submission FAILED validation (falling back)")

    print("\n=== Variant summary ===")
    print(f"  variant: {variant_name} ({vdef['label']})  min_track_len={cfg['min_track_len']}  edge_policy_veto={cfg['edge_policy_veto']}")
    print(f"  artifact: {selection['artifact_name']}  guard_passed: {guard_passed}  sha: {str(selection['weight_sha256'])[:12]}")
    print(f"  raw: {conversion['raw_nodes']}/{conversion['raw_edges']}  final: {inv['n_node_rows']}/{inv['n_edge_rows']}")
    print(f"  gap_synth={synthetic} div+={conversion['safe_divisions_added']} veto={conversion['repair_policy_edges_vetoed']} "
          f"short_rm={conversion['short_track_nodes_removed']} linefit={conversion['linefit_smoothed_nodes']}")
    print(f"  topology: in<={inv['max_in_degree']} out<={inv['max_out_degree']} multiframe={inv['direct_multiframe_edges']} dangling={inv['dangling_edges']}")
    print(f"  RECOMMENDATION: {recommendation}")
    log("submission.csv built and validated. Do not submit until user reviews (M27 experimental).")
    return {"submission_df": submission_df, "variant": variant_name, "valid": valid,
            "recommendation": recommendation, "report": report_obj}


def _m27_base_report(variant_name, vdef, selection, guard_passed, inv, conversion, valid, fallback_used,
                     final_source, recommendation, instruction) -> dict:
    cfg = vdef["config"]
    c = conversion or {}
    i = inv or {}
    return {
        "variant": variant_name, "label": vdef["label"],
        "artifact_name": selection.get("artifact_name"), "weight_sha256": selection.get("weight_sha256"),
        "artifact_guard_passed": guard_passed,
        "raw_nodes": c.get("raw_nodes", 0), "raw_edges": c.get("raw_edges", 0),
        "final_nodes": c.get("final_nodes", 0), "final_edges": c.get("final_edges", 0),
        "gap_candidates": c.get("gap_candidates", 0), "gap_pairs_selected": c.get("gap_pairs_selected", 0),
        "gap_inserted_synthetic": c.get("gap_inserted_synthetic", 0), "gap_added_edges": c.get("gap_added_edges", 0),
        "safe_division_candidates": c.get("safe_division_candidates", 0), "safe_divisions_added": c.get("safe_divisions_added", 0),
        "repair_policy_edge_candidates": c.get("repair_policy_edge_candidates", 0),
        "repair_policy_edges_proposed_veto": c.get("repair_policy_edges_proposed_veto", 0),
        "repair_policy_edges_vetoed": c.get("repair_policy_edges_vetoed", 0),
        "repair_policy_edges_capped": c.get("repair_policy_edges_capped", 0),
        "short_track_components_removed": c.get("short_track_components_removed", 0),
        "short_track_nodes_removed": c.get("short_track_nodes_removed", 0),
        "short_track_edges_removed": c.get("short_track_edges_removed", 0),
        "linefit_smoothed_nodes": c.get("linefit_smoothed_nodes", 0),
        "n_node_rows": i.get("n_node_rows", 0), "n_edge_rows": i.get("n_edge_rows", 0),
        "max_in_degree": i.get("max_in_degree", 0), "max_out_degree": i.get("max_out_degree", 0),
        "direct_multiframe_edges": i.get("direct_multiframe_edges", 0), "dangling_edges": i.get("dangling_edges", 0),
        "id_consecutive": i.get("id_consecutive", None), "no_nan": i.get("no_nan", None),
        "valid": valid, "fallback_used": fallback_used, "final_source": final_source,
        "recommendation": recommendation,
        "min_track_len": cfg["min_track_len"], "edge_policy_veto_enabled": cfg["edge_policy_veto"],
        "keep_division_components": cfg["keep_division_components"],
        "missing_datasets": i.get("missing_datasets", []), "emptied_datasets": i.get("emptied_datasets", []),
        "synthetic_gap_cap": M27_MAX_SYNTHETIC_GAP,
        "public_reference": {"raw_nodes": 127790, "raw_edges": 115694, "final_node_rows": 119763,
                             "final_edge_rows": 115080, "submission_rows": 234843,
                             "artifact_name": "biohub-tracking-support-pack-400ep-snapshot-v1",
                             "weight_sha256": M27_400EP_SHA256},
        "final_instruction": instruction,
    }


def run_m27_variant_with_fallback(
    variant_name: str, working_dir: str = KAGGLE_WORKING_DIR, competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
) -> dict:
    _RUN_LOG.clear()
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    sample_submission_path = Path(competition_dir) / "sample_submission.csv"
    try:
        result = run_m27_variant_pipeline(variant_name, working_dir, competition_dir)
        _write_log(out_dir)
        return {"status": "ok", **result}
    except Exception:
        tb = traceback.format_exc()
        log("[error] M27 variant pipeline failed:\n" + tb)
        if submission_path.exists():
            log("submission.csv already exists; keeping it (not overwriting with fallback).")
            (out_dir / "milestone27_failure_fallback_report.json").write_text(json.dumps(
                {"variant": variant_name, "traceback": tb, "fallback": {"used": False, "reason": "submission already written"}}, indent=2))
            _write_log(out_dir)
            return {"status": "pipeline_failed_submission_exists"}
        try:
            fb = write_fallback_submission(sample_submission_path, submission_path)
            (out_dir / "milestone27_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback": fb}, indent=2))
            print("\n=== Variant summary ===")
            print(f"  variant: {variant_name}  final_source: fallback_sample_submission  fallback_used: True")
            print("  RECOMMENDATION: DO_NOT_SUBMIT_VALIDATION_FAILED")
            log("FALLBACK sample submission written because the M27 pipeline failed.")
            _write_log(out_dir)
            return {"status": "fallback_written", "fallback": fb}
        except Exception:
            fb_tb = traceback.format_exc()
            try:
                (out_dir / "milestone27_failure_fallback_report.json").write_text(json.dumps({"variant": variant_name, "traceback": tb, "fallback_error": fb_tb}, indent=2))
            except Exception:
                pass
            log("[fatal] could not write fallback submission either:\n" + fb_tb)
            _write_log(out_dir)
            raise

# --------------------------------------------------------------------------- #
# 45. M27 tests + top-level driver
# --------------------------------------------------------------------------- #
def _mk27_nodes(rows):
    return pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x", "score"])


def _mk27_edges(rows):
    return pd.DataFrame(rows, columns=["source_id", "target_id", "edge_prob", "edge_dist"])


def _m27_test_motion_relink():
    # Two well-separated straight tracks; relink must keep each track and never
    # cross-link (cross distance >> relaxed), giving in<=1/out<=1 unit edges.
    nodes = _mk27_nodes([(t, t, 0.0, float(t), 0.0, 0.9) for t in range(4)] +
                        [(10 + t, t, 0.0, 100.0 + t, 0.0, 0.9) for t in range(4)])
    edges = _mk27_edges([(t, t + 1, 0.9, 0.41) for t in range(3)] +
                        [(10 + t, 11 + t, 0.9, 0.41) for t in range(3)])
    relinked, diag = motion_relink_edges(nodes, edges, 6.0, 10.0, 0.5, 0.75)
    assert len(relinked) == 6, f"expected 6 relinked edges, got {len(relinked)}"
    t_of = dict(zip(nodes["node_id"], nodes["t"]))
    for r in relinked.itertuples():
        assert t_of[int(r.target_id)] - t_of[int(r.source_id)] == 1, "relink produced a non-unit edge"
    assert int(relinked.groupby("target_id").size().max()) <= 1 and int(relinked.groupby("source_id").size().max()) <= 1


def _m27_test_gap_close_and_reuse():
    # END@t1 and START@t3 with a missing t2 -> one bridge.
    nodes = _mk27_nodes([(0, 0, 0.0, 0.0, 0.0, 0.9), (1, 1, 0.0, 1.0, 0.0, 0.9),
                         (2, 3, 0.0, 3.0, 0.0, 0.9), (3, 4, 0.0, 4.0, 0.0, 0.9)])
    edges = pd.DataFrame([(0, 1), (2, 3)], columns=["source_id", "target_id"])
    n2, e2, d, _ = close_single_frame_gaps_m27(nodes, edges, 6.0, True, 3.2, 0.05, 2000)
    assert d["gap_pairs_selected"] == 1 and d["gap_inserted_synthetic"] == 1 and d["gap_reused_existing"] == 0, d
    t_of = dict(zip(n2["node_id"], n2["t"]))
    for r in e2.itertuples():
        assert t_of[int(r.target_id)] - t_of[int(r.source_id)] == 1, "gap close produced a non-unit edge"
    # Same but with an existing free node at the midpoint -> reuse (no synthetic).
    nodes_r = pd.concat([nodes, _mk27_nodes([(4, 2, 0.0, 2.0, 0.0, 0.2)])], ignore_index=True)
    n3, e3, d3, _ = close_single_frame_gaps_m27(nodes_r, edges, 6.0, True, 3.2, 0.05, 2000)
    assert d3["gap_pairs_selected"] == 1 and d3["gap_reused_existing"] == 1 and d3["gap_inserted_synthetic"] == 0, d3


def _m27_test_safe_divisions():
    nodes = _mk27_nodes([(0, 0, 0.0, 0.0, 0.0, 0.9), (1, 1, 0.0, 2.0, 0.0, 0.9), (2, 1, 0.0, 1.0, 0.0, 0.8)])
    edges = pd.DataFrame([(0, 1)], columns=["source_id", "target_id"])
    _n, e2, d = add_safe_divisions_postlink(nodes, edges, 4.7, 7.2, 7.8, 0.008, 0.004)
    assert d["safe_divisions_added"] == 1, d
    assert int(e2.groupby("source_id").size().max()) == 2, "source should now be a division (out_degree 2)"


def _m27_test_edge_policy_veto():
    # Long strong chain (gives the 1% cap headroom) + 5 weak edges from
    # dedicated non-chain sources at t0. Source 300 is ALSO a division (extra
    # strong edge) so its weak edge must be skipped; 301..304 are eligible.
    N = 200
    node_rows = [(i, i, 0.0, float(i), 0.0, 0.9) for i in range(N)]
    edge_rows = [(i, i + 1, 0.95, 0.41) for i in range(N - 1)]
    for k in range(5):                       # weak sources 300..304 at t0 -> bad targets at t1
        node_rows.append((300 + k, 0, 0.0, 0.0, 0.0, 0.2))
        node_rows.append((1000 + k, 1, 0.0, 60.0, 0.0, 0.1))
        edge_rows.append((300 + k, 1000 + k, 0.0, 24.0))     # weak: prob 0, long distance
    node_rows.append((1100, 1, 0.0, 0.4, 0.0, 0.9))          # near 300 -> strong second child
    edge_rows.append((300, 1100, 0.9, 0.41))                 # makes 300 a division
    nodes = _mk27_nodes(node_rows)
    edges = _mk27_edges(edge_rows)
    surv, scores, diag = apply_edge_policy_veto(nodes, edges, 0.35, 0.01, True)
    cap = int(0.01 * len(edges))
    assert cap >= 1, cap
    assert diag["repair_policy_edges_proposed_veto"] == 4, diag        # 300's weak skipped (division)
    assert diag["repair_policy_edges_vetoed"] == cap, (diag, cap)
    assert diag["repair_policy_edges_capped"] == 4 - cap, diag
    # the division-source (300) weak edge 300->1000 must be skipped (survives).
    assert ((surv["source_id"] == 300) & (surv["target_id"] == 1000)).any(), "division-source weak edge must be skipped"


def _m27_test_short_track_filter():
    # comp1: 3-node chain (short); comp2: 8-node chain (long); comp3: 2-node
    # component containing a division (kept when keep_division_components).
    rows = []
    edges = []
    for i in range(3):          # comp1 ids 0..2
        rows.append((i, i, 0.0, float(i), 0.0, 0.9))
    edges += [(0, 1), (1, 2)]
    for k in range(8):          # comp2 ids 10..17
        rows.append((10 + k, k, 0.0, 50.0 + k, 0.0, 0.9))
    edges += [(10 + k, 11 + k) for k in range(7)]
    # comp3: division at id 20 (t0) -> 21,22 (t1) ids, 3 nodes but has a division.
    rows += [(20, 0, 0.0, 90.0, 0.0, 0.9), (21, 1, 0.0, 90.0, 0.0, 0.9), (22, 1, 0.0, 91.0, 0.0, 0.9)]
    edges += [(20, 21), (20, 22)]
    nodes = _mk27_nodes(rows)
    edf = pd.DataFrame(edges, columns=["source_id", "target_id"])
    n2, e2, d = filter_short_track_components(nodes, edf, 7, True)
    ids = set(n2["node_id"])
    assert {0, 1, 2}.isdisjoint(ids), "short non-division component should be removed"
    assert {10, 11, 17}.issubset(ids), "long component should be kept"
    assert {20, 21, 22}.issubset(ids), "short DIVISION component should be kept (keep_division_components)"
    assert d["short_track_components_removed"] == 1 and d["short_track_nodes_removed"] == 3, d


def _m27_test_final_safety_repair():
    # source 0 has 3 children at t1 (out_degree 3); one non-unit edge; one
    # dangling edge; one NaN node.
    nodes = _mk27_nodes([(0, 0, 0.0, 0.0, 0.0, 0.9), (1, 1, 0.0, 1.0, 0.0, 0.9), (2, 1, 0.0, 1.1, 0.0, 0.9),
                         (3, 1, 0.0, 1.2, 0.0, 0.9), (4, 2, 0.0, 2.0, 0.0, 0.9),
                         (5, 1, 0.0, float("nan"), 0.0, 0.9)])
    edges = _mk27_edges([(0, 1, 0.9, 0.41), (0, 2, 0.8, 0.45), (0, 3, 0.7, 0.49),
                         (0, 4, 0.95, 0.82), (0, 99, 0.9, 0.4), (0, 5, 0.9, 0.4)])
    n2, e2, d = final_safety_repair(nodes, edges)
    assert 5 not in set(n2["node_id"]), "NaN node must be dropped"
    t_of = dict(zip(n2["node_id"], n2["t"]))
    for r in e2.itertuples():
        assert int(r.source_id) in t_of and int(r.target_id) in t_of, "dangling edge survived"
        assert t_of[int(r.target_id)] - t_of[int(r.source_id)] == 1, "non-unit edge survived"
    assert int(e2.groupby("source_id").size().max()) <= 2, "out_degree>2 survived"
    assert int(e2.groupby("target_id").size().max()) <= 1, "in_degree>1 survived"


def _m27_test_end_to_end_submission():
    # A clean 3-track fixture through the full pipeline -> valid submission.
    rows, edges = [], []
    for tr in range(3):
        base = tr * 100
        for t in range(9):
            rows.append((base + t, t, 0.0, float(base + t), 0.0, 0.9))
        edges += [(base + t, base + t + 1, 0.9, 0.41) for t in range(8)]
    nodes = _mk27_nodes(rows)
    edf = _mk27_edges(edges)
    cfg = _m27_cfg(min_track_len=3)
    pn, pe, diag = apply_public_pipeline(nodes, edf, cfg)
    pn, pe, _ = relabel_positive(pn, pe, 0)
    assert len(pn) > 0 and len(pe) > 0, "pipeline emptied a clean fixture"
    all_rows, nid = [], 0
    for r in pn.itertuples():
        all_rows.append({"id": nid, "dataset": "ds1", "row_type": "node", "node_id": int(r.node_id),
                         "t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x), "source_id": -1, "target_id": -1})
        nid += 1
    for r in pe.itertuples():
        all_rows.append({"id": nid, "dataset": "ds1", "row_type": "edge", "node_id": -1,
                         "t": -1, "z": -1, "y": -1, "x": -1, "source_id": int(r.source_id), "target_id": int(r.target_id)})
        nid += 1
    df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    valid, rep, stats = validate_reference_submission(df, ["ds1"])
    assert valid, f"end-to-end submission must be valid: {rep}"
    inv = _m27_submission_invariants(df, ["ds1"])
    assert inv["direct_multiframe_edges"] == 0 and inv["dangling_edges"] == 0 and inv["max_in_degree"] <= 1 and inv["max_out_degree"] <= 2


def _m27_test_variant_registry_and_guard():
    assert M27_VARIANTS["A"]["config"]["min_track_len"] == 7 and M27_VARIANTS["A"]["config"]["edge_policy_veto"] is True
    assert M27_VARIANTS["B"]["config"]["edge_policy_veto"] is False and M27_VARIANTS["B"]["config"]["min_track_len"] == 7
    assert M27_VARIANTS["C"]["config"]["min_track_len"] == 5 and M27_VARIANTS["C"]["config"]["edge_policy_veto"] is True
    assert M27_VARIANTS["D"]["config"]["min_track_len"] == 8 and M27_VARIANTS["D"]["config"]["keep_division_components"] is True
    # guard needs BOTH the 400ep name AND the exact sha.
    import tempfile
    with tempfile.TemporaryDirectory() as tdroot:
        root = Path(tdroot) / "biohub-tracking-support-pack-400ep-snapshot-v1"
        (root / "repo" / "scripts").mkdir(parents=True)
        (root / "repo" / "scripts" / "predict_unet_transformer.py").write_text("# predict\n")
        (root / "weights" / "unet_transformer" / "split_0").mkdir(parents=True)
        (root / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth").write_bytes(b"not-the-real-weights")
        (root / "ARTIFACT_MANIFEST.json").write_text(json.dumps({"artifact_name": "biohub-tracking-support-pack-400ep-snapshot-v1"}))
        info = inspect_400ep_candidate(root)
        assert info["name_has_400ep"] is True and info["sha_matches_400ep"] is False and info["is_400ep_artifact"] is False, info
        other = Path(tdroot) / "some-other-pack"
        (other / "repo" / "scripts").mkdir(parents=True)
        (other / "repo" / "scripts" / "predict_unet_transformer.py").write_text("# predict\n")
        (other / "weights" / "unet_transformer" / "split_0").mkdir(parents=True)
        (other / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth").write_bytes(b"x")
        assert inspect_400ep_candidate(other)["name_has_400ep"] is False


def _m27_test_recommendation_gates():
    ok_inv = {"n_node_rows": 120000, "n_edge_rows": 115000, "max_in_degree": 1, "max_out_degree": 2,
              "direct_multiframe_edges": 0, "dangling_edges": 0, "id_consecutive": True, "no_nan": True,
              "missing_datasets": [], "emptied_datasets": []}
    assert _m27_recommendation("A", True, True, False, ok_inv, 2100) == "OK_TO_SUBMIT_EXPERIMENTAL"
    assert _m27_recommendation("A", False, True, False, ok_inv, 2100) == "DO_NOT_SUBMIT_WRONG_ARTIFACT"
    assert _m27_recommendation("A", True, True, False, {**ok_inv, "dangling_edges": 3}, 2100) == "DO_NOT_SUBMIT_VALIDATION_FAILED"
    assert _m27_recommendation("A", True, True, False, {**ok_inv, "n_node_rows": 5000}, 2100) == "DO_NOT_SUBMIT_UNSANE_NODE_COUNT"
    assert _m27_recommendation("A", True, True, False, ok_inv, 3000) == "DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION"
    assert _m27_recommendation("C", True, True, False, {**ok_inv, "n_node_rows": 127000}, 2100) == "DO_NOT_SUBMIT_MINLEN5_NODES_TOO_HIGH"
    assert _m27_recommendation("D", True, True, False, {**ok_inv, "n_node_rows": 111000}, 2100) == "DO_NOT_SUBMIT_STRICT_PRECISION_UNDERCUT"
    assert _m27_recommendation("A", True, True, False, {**ok_inv, "emptied_datasets": ["ds9"]}, 2100) == "DO_NOT_SUBMIT_EMPTY_OR_MISSING_DATASET"


def run_milestone27_tests() -> None:
    """Public-notebook pipeline math on synthetic fixtures (no Kaggle/GPU):
    motion relink, gap close + reuse, safe divisions, edge-policy veto (with
    cap + division-source skip), short-track filtering (keep divisions), final
    safety repair (degree/unit-t/dangling/NaN), an end-to-end valid submission,
    the variant registry, the 400ep guard decomposition, and the recommendation
    gates."""
    _m27_test_motion_relink()
    _m27_test_gap_close_and_reuse()
    _m27_test_safe_divisions()
    _m27_test_edge_policy_veto()
    _m27_test_short_track_filter()
    _m27_test_final_safety_repair()
    _m27_test_end_to_end_submission()
    _m27_test_variant_registry_and_guard()
    _m27_test_recommendation_gates()
    print("All milestone27_public_notebook_repro_runner tests passed (9/9).")


DEFAULT_M27_VARIANT = "A"   # submit-order first: faithful public reproduction + safety


def run_milestone27_variant(
    variant_name: str = DEFAULT_M27_VARIANT, working_dir: str = KAGGLE_WORKING_DIR,
    competition_dir: str = KAGGLE_COMPETITION_INPUT_DIR,
) -> dict:
    print("=== Self-test: M27 public-notebook pipeline (relink, gap, divisions, veto, short-track, safety) ===")
    run_milestone27_tests()
    if variant_name not in M27_VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}, expected one of {sorted(M27_VARIANTS)}")
    if not is_kaggle_env():
        print(f"[dry-run] /kaggle/input absent - self-tests only (variant {variant_name}). On Kaggle this guards the "
              "400ep artifact, runs the reference predict, reproduces the public output pipeline, and writes a guarded submission.")
        return {"status": "dry_run", "variant": variant_name}
    return run_m27_variant_with_fallback(variant_name, working_dir, competition_dir)


if __name__ == "__main__":
    run_milestone27_variant(DEFAULT_M27_VARIANT)
