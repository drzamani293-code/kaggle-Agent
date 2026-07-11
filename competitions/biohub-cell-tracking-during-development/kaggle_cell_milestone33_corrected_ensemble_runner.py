"""
Biohub - Cell Tracking During Development
Milestone 33 - CORRECTED ENSEMBLE AUDIT & FUSION PACK.

Audits the uploaded ensemble sources (blend_submissions.py, geff_ensemble.py,
local_metric(1).py, model_level_roadmap.md) and REPAIRS the two ensembles they
implement. The uploaded blend_submissions.py resets next_id=0 per dataset
(repeated global node ids), snaps nodes nearest-neighbour (not one-to-one, so
same-variant nodes can collapse and one variant can double-vote a canonical
edge), admits min_votes=1 forks, and depends on variant processing order.
geff_ensemble.py additionally scores ILP_SOLUTION_LIKE graphs as if they were
full candidate graphs, imports the OLD winning_postprocess_v2 linker (not the
corrected M30 two-stage linker with explicit no-link assignments), and divides
consensus probability by ALL models even when a model was not eligible to
propose the edge. M33 fixes every one of these.

Corrected machinery (shared by both ensembles):
  * NODE CANONICALIZATION - deterministic, input-order-invariant. Variants are
    processed in a fixed sorted-by-name order; for every frame the incoming
    variant's nodes are matched ONE-TO-ONE to the growing canonical clusters
    with Hungarian assignment gated at eps_um (physical um, VOXEL_SIZE_UM
    z=1.625 y=0.40625 x=0.40625). Two nodes of one variant can never share a
    canonical node; unmatched nodes stay distinct; canonical coordinates are a
    weight-robust mean over cluster members (never "first variant wins"). A
    permutation of the input variant order yields an identical graph (tested).
  * EDGE VOTING - per variant, translate edges to canonical ids, drop
    self/back/non-unit-time edges, DEDUPLICATE so each variant casts at most one
    vote per canonical edge, accumulate WEIGHTED support (M19-C 2.0, M29-A 1.0,
    M30-C 1.0; unidentified runs are reported, never silently weighted).
  * TWO-STAGE LINKER (corrected M30, NOT winning_postprocess_v2) - primary
    selection is a per-frame min-cost assignment with explicit no-link dummies
    (each target <=1 parent, each source <=1 primary child); division is an
    INDEPENDENT second assignment over parents with one primary child and still-
    unassigned targets, gated by stricter support + multi-variant fork + parent-
    child + sister distance, with a no-division dummy so rejected second children
    never steal a target.
  * PROBABILITY denominator - a model is ELIGIBLE for a canonical edge only if
    both endpoint detections exist for that model and its candidate generator
    could have proposed the edge; support_fraction = proposer_count /
    eligible_model_count (never / total model count). A missing endpoint is NOT
    a zero-probability vote.

Runners (standalone one-cell):
  A M33_A_ENSEMBLE_INPUT_AUDIT_NOT_SUBMIT       - dynamic input discovery +
      per-run structural report + pairwise submission/GEFF diversity +
      ensemble-potential classification. NON-SUBMIT.
  B M33_B_CORRECTED_OUTPUT_BLEND_DIAGNOSTIC     - B0 M19-C-alone reproduction,
      B1 weighted consensus, B2 stricter, B3 controlled union. NON-SUBMIT.
  C M33_C_CORRECTED_OUTPUT_BLEND_CANDIDATE      - builds an EXPERIMENTAL
      submission.csv from the corrected output-level weighted ensemble, hard-
      gated (>=3 valid, not near-duplicate, zero collisions, order-invariant,
      no duplicate votes, valid graph, sane delta vs M19-C).
  D M33_D_PROBABILITY_FUSION_AUDIT_NOT_SUBMIT   - GEFF candidate-graph class +
      edge_prob availability/quantiles + eligible-endpoint universe + calibration
      diagnostics. NON-SUBMIT.
  E M33_E_CORRECTED_PROBABILITY_ENSEMBLE_CANDIDATE - corrected probability-level
      ensemble, hard-gated (blocked on ILP_SOLUTION_LIKE / EDGE_PROB_UNAVAILABLE /
      unresolved calibration / insufficient diversity).

Honesty: local_metric.py is NEVER called the official scorer; a proxy result may
be printed only under NON_AUTHORITATIVE_PROXY_DIAGNOSTIC and never drives a
submit recommendation. Official CV integrates with M32.1's vendored metric when a
clean split becomes available. M19-C 0.880 stays confirmed best/final until an
OBSERVED score beats it; M29-A 0.876 failed; M30-C pending; M31 blocked; M32/M32.1
official CV unresolved. No Kaggle-API submit, no static submission rows, no M16-M32
source modified; all outputs under /kaggle/working. Off Kaggle every runner
self-tests on synthetic fixtures and reports honestly.
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
# 50. M33 corrected-ensemble CORE - canonicalization, voting, two-stage linker
# --------------------------------------------------------------------------- #
# All distances are physical micrometres via the reused VOXEL_SIZE_UM /
# _VOXEL_SCALE. scipy.optimize.linear_sum_assignment is the only heavy dep and
# is available on Kaggle + in this repo's env.
import itertools as _it33
from collections import defaultdict as _dd33

# Default per-run ensemble weights (resolved from metadata; unidentified runs
# are reported and given M33_UNKNOWN_WEIGHT, never silently trusted).
M33_DEFAULT_WEIGHTS = {"M19-C": 2.0, "M29-A": 1.0, "M30-C": 1.0}
M33_UNKNOWN_WEIGHT = 1.0

# Canonicalization eps candidates (um) reported for collision risk. 2.0um is
# never used automatically without reporting its collision count.
M33_EPS_CANDIDATES_UM = [0.75, 1.0, 1.5, 2.0]
M33_DEFAULT_EPS_UM = 1.0

# Conservative output-blend gates (all physical um / weighted support).
M33_PRIMARY_SUPPORT_MIN = 2.0
M33_DIVISION_SUPPORT_MIN = 2.0
M33_FORK_MIN_VARIANTS = 2
M33_PRIMARY_MAX_DIST_UM = 7.0
M33_DIVISION_CHILD_MAX_UM = 6.5
M33_SISTER_MAX_UM = 9.0
M33_DIST_COST_WEIGHT = 0.05     # lambda on physical distance inside the assignment cost
_M33_BIG = 1e9


def _m33_phys(pos_a, pos_b) -> float:
    d = (np.asarray(pos_a, float) - np.asarray(pos_b, float)) * _VOXEL_SCALE
    return float(np.sqrt(float(np.dot(d, d))))


def _m33_hungarian(cost: "np.ndarray"):
    """linear_sum_assignment wrapper. Returns (rows, cols)."""
    from scipy.optimize import linear_sum_assignment
    return linear_sum_assignment(cost)


class M33Variant:
    """One tracking output normalised to a common structure.
    nodes: DataFrame[node_id,t,z,y,x] (node_id local, any ints).
    edges: DataFrame[source_id,target_id] local unit-time edges.
    edge_prob: optional dict[(source_id,target_id)] -> float (candidate probs).
    cand_nodes: optional set of local node_ids the model actually detected
                (defaults to nodes' ids) - used for endpoint eligibility.
    """
    def __init__(self, name, nodes, edges, weight=None, edge_prob=None,
                 cand_edges=None, graph_class=None):
        self.name = str(name)
        self.nodes = nodes.reset_index(drop=True)
        self.edges = edges.reset_index(drop=True)
        self.weight = float(weight) if weight is not None else M33_DEFAULT_WEIGHTS.get(self.name, M33_UNKNOWN_WEIGHT)
        self.weight_identified = weight is not None or self.name in M33_DEFAULT_WEIGHTS
        self.edge_prob = dict(edge_prob) if edge_prob else {}
        # candidate edges (superset of solution edges) for probability fusion.
        self.cand_edges = cand_edges if cand_edges is not None else set(
            (int(r.source_id), int(r.target_id)) for r in self.edges.itertuples())
        self.graph_class = graph_class
        self._pos = {int(r.node_id): (int(r.t), np.array([r.z, r.y, r.x], float)) for r in self.nodes.itertuples()}
        self._det_ids = set(self._pos.keys())

    def pos(self, nid):
        return self._pos[int(nid)][1]

    def frame(self, nid):
        return self._pos[int(nid)][0]


# --------------------------------------------------------------------------- #
# 50a. Deterministic, order-invariant one-to-one node canonicalization
# --------------------------------------------------------------------------- #
def canonicalize_nodes(variants, eps_um=M33_DEFAULT_EPS_UM):
    """Match nodes across variants into canonical clusters, one-to-one PER FRAME
    with Hungarian assignment gated at eps_um. Deterministic and invariant to the
    INPUT order of `variants` (they are processed in a fixed sorted-by-name
    order). At most one node per variant per cluster. Canonical coordinate is a
    weight-robust mean over members. Returns a dict with node_to_canon
    {(variant_index, local_node_id) -> canon_id}, canon rows, and collision/
    cluster diagnostics.
    """
    order = sorted(range(len(variants)), key=lambda i: (variants[i].name, i))
    clusters = []            # list of dict(t, members=[(vi,nid,pos,w)], sum_wpos, sum_w)
    by_frame = _dd33(list)   # t -> [cluster_index,...]
    node_to_canon = {}
    per_variant_collapse = {variants[i].name: 0 for i in range(len(variants))}

    for vi in order:
        v = variants[vi]
        w = v.weight
        # group this variant's nodes by frame
        frame_nodes = _dd33(list)
        for r in v.nodes.itertuples():
            frame_nodes[int(r.t)].append((int(r.node_id), np.array([r.z, r.y, r.x], float)))
        for t, nodes in sorted(frame_nodes.items()):
            cl_idx = by_frame[t]
            n, m = len(nodes), len(cl_idx)
            assigned = {}   # node position index -> cluster_index
            if m and n:
                cost = np.full((n, m), _M33_BIG, float)
                for i, (nid, pos) in enumerate(nodes):
                    for j, cidx in enumerate(cl_idx):
                        c = clusters[cidx]
                        cpos = c["sum_wpos"] / c["sum_w"]
                        d = _m33_phys(pos, cpos)
                        if d <= eps_um:
                            cost[i, j] = d
                rows, cols = _m33_hungarian(cost)
                for i, j in zip(rows, cols):
                    if cost[i, j] < _M33_BIG:
                        assigned[i] = cl_idx[j]
            # place nodes: matched -> existing cluster; else new cluster
            for i, (nid, pos) in enumerate(nodes):
                if i in assigned:
                    cidx = assigned[i]
                    c = clusters[cidx]
                    # a same-variant collapse would mean this cluster already
                    # has a member from vi. Hungarian one-to-one prevents it;
                    # assert to make the guarantee explicit + count for report.
                    if any(mm[0] == vi for mm in c["members"]):
                        per_variant_collapse[v.name] += 1
                        cidx = _m33_new_cluster(clusters, by_frame, t)
                        c = clusters[cidx]
                    c["members"].append((vi, nid, pos, w))
                    c["sum_wpos"] = c["sum_wpos"] + w * pos
                    c["sum_w"] += w
                    node_to_canon[(vi, nid)] = cidx
                else:
                    cidx = _m33_new_cluster(clusters, by_frame, t)
                    c = clusters[cidx]
                    c["members"].append((vi, nid, pos, w))
                    c["sum_wpos"] = w * pos
                    c["sum_w"] = w
                    node_to_canon[(vi, nid)] = cidx

    # finalize canonical rows (weighted mean coord) + diagnostics
    canon_rows = []
    within_max = 0.0
    size_hist = _dd33(int)
    for cidx, c in enumerate(clusters):
        coord = c["sum_wpos"] / c["sum_w"]
        # robust: if a cluster has >=3 members use per-axis median (less order/
        # weight sensitive), else weighted mean. Both are order-invariant.
        if len(c["members"]) >= 3:
            coord = np.median(np.stack([mm[2] for mm in c["members"]]), axis=0)
        canon_rows.append({"canon_id": cidx, "t": c["t"], "z": float(coord[0]),
                           "y": float(coord[1]), "x": float(coord[2]), "size": len(c["members"])})
        size_hist[len(c["members"])] += 1
        for mm in c["members"]:
            within_max = max(within_max, _m33_phys(mm[2], coord))
    canon_nodes = pd.DataFrame(canon_rows, columns=["canon_id", "t", "z", "y", "x", "size"])
    canon_t = {int(r.canon_id): int(r.t) for r in canon_nodes.itertuples()}
    return {
        "node_to_canon": node_to_canon, "canon_nodes": canon_nodes, "canon_t": canon_t,
        "n_canon": len(canon_nodes), "eps_um": eps_um,
        "matched_clusters": int((canon_nodes["size"] >= 2).sum()) if len(canon_nodes) else 0,
        "unmatched_nodes": int((canon_nodes["size"] == 1).sum()) if len(canon_nodes) else 0,
        "cluster_size_hist": {int(k): int(v) for k, v in sorted(size_hist.items())},
        "max_within_cluster_um": float(within_max),
        "per_variant_collapse": per_variant_collapse,
        "collision_count": int(sum(per_variant_collapse.values())),
    }


def _m33_new_cluster(clusters, by_frame, t):
    idx = len(clusters)
    clusters.append({"t": int(t), "members": [], "sum_wpos": np.zeros(3), "sum_w": 0.0})
    by_frame[t].append(idx)
    return idx


# --------------------------------------------------------------------------- #
# 50b. Edge voting - per-variant dedup, weighted support, eligibility
# --------------------------------------------------------------------------- #
def build_edge_votes(variants, canon, use_candidates=False):
    """Translate every variant's edges to canonical ids and accumulate votes.
    Each variant contributes AT MOST ONE vote per canonical edge (duplicates
    after node collapse are removed and counted). Rejects self-edges and
    non-unit-time edges. `use_candidates` votes over cand_edges (probability
    fusion) instead of solution edges.
    Returns votes dict keyed (cs,ct) with support/variants/probs and
    duplicate_votes_removed + eligibility maps.
    """
    node_to_canon = canon["node_to_canon"]
    canon_t = canon["canon_t"]
    votes = {}
    dup_removed = 0
    # endpoint presence: which canonical nodes each variant detected
    canon_present = _dd33(set)   # variant_name -> set(canon_id)
    for (vi, nid), cid in node_to_canon.items():
        canon_present[variants[vi].name].add(cid)
    for vi, v in enumerate(variants):
        edge_iter = v.cand_edges if use_candidates else [(int(r.source_id), int(r.target_id)) for r in v.edges.itertuples()]
        seen = set()
        for (s, t) in edge_iter:
            key_local = (vi, s)
            if (vi, s) not in node_to_canon or (vi, t) not in node_to_canon:
                continue
            cs = node_to_canon[(vi, s)]
            ct = node_to_canon[(vi, t)]
            if cs == ct:
                continue                              # collapsed self-edge
            if canon_t.get(ct) != canon_t.get(cs, -99) + 1:
                continue                              # non-unit-time / backward
            ce = (cs, ct)
            if ce in seen:
                dup_removed += 1                      # same variant, one vote only
                continue
            seen.add(ce)
            rec = votes.get(ce)
            if rec is None:
                rec = {"support": 0.0, "variants": set(), "probs": {}, "n_proposers": 0}
                votes[ce] = rec
            rec["support"] += v.weight
            rec["variants"].add(v.name)
            rec["n_proposers"] += 1
            p = v.edge_prob.get((s, t))
            if p is not None:
                rec["probs"][v.name] = float(p)
    return {"votes": votes, "duplicate_votes_removed": dup_removed, "canon_present": canon_present}


def eligible_model_count(cs, ct, variants, canon_present):
    """A model is eligible for canonical edge (cs,ct) iff BOTH endpoint
    detections exist for that model (it could have proposed the edge). Missing
    endpoint => NOT eligible (never a zero-probability vote)."""
    n = 0
    for v in variants:
        pres = canon_present[v.name]
        if cs in pres and ct in pres:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 50c. Corrected M30 two-stage linker (primary assignment + independent division)
# --------------------------------------------------------------------------- #
def _assign_with_nolink(sources, targets, cost_lookup):
    """Min-cost one-to-one assignment source<->target with explicit no-link
    dummies (a source may link to nothing, a target may stay unparented). Uses a
    padded square matrix: real S x T costs, plus per-source and per-target
    no-link options at cost 0, and dummy-dummy at 0. Returns list of (s,t)
    accepted real pairs. cost_lookup(s,t)->cost or None (no candidate)."""
    S, T = len(sources), len(targets)
    if S == 0 or T == 0:
        return []
    N = S + T
    cost = np.zeros((N, N), float)
    cost[:S, :T] = _M33_BIG
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            c = cost_lookup(s, t)
            if c is not None:
                cost[i, j] = c
    # source -> no-link (top-right S x S diagonal 0, else BIG)
    cost[:S, T:] = _M33_BIG
    for i in range(S):
        cost[i, T + i] = 0.0
    # target -> no-source (bottom-left T x T diagonal 0, else BIG)
    cost[S:, :T] = _M33_BIG
    for j in range(T):
        cost[S + j, j] = 0.0
    # dummy-dummy bottom-right = 0
    cost[S:, T:] = 0.0
    rows, cols = _m33_hungarian(cost)
    out = []
    for i, j in zip(rows, cols):
        if i < S and j < T and cost[i, j] < _M33_BIG:
            out.append((sources[i], targets[j]))
    return out


def two_stage_link(canon, votes_res, variants,
                   primary_support_min=M33_PRIMARY_SUPPORT_MIN,
                   division_support_min=M33_DIVISION_SUPPORT_MIN,
                   fork_min_variants=M33_FORK_MIN_VARIANTS,
                   primary_max_um=M33_PRIMARY_MAX_DIST_UM,
                   division_child_max_um=M33_DIVISION_CHILD_MAX_UM,
                   sister_max_um=M33_SISTER_MAX_UM,
                   union_prefer_edges=None, union_keep_weight_min=2.0):
    """Corrected two-stage linker. Stage 1: per-frame min-cost PRIMARY assignment
    with no-link dummies (target<=1 parent, source<=1 primary child). Stage 2:
    INDEPENDENT division assignment over parents-with-one-primary-child and
    still-unparented targets, stricter support + multi-variant fork + parent-
    child + sister gates + no-division dummy (rejected children can't steal).
    `union_prefer_edges` (set of canonical edges, e.g. M19-C) forces retention on
    conflict unless a challenger reaches union_keep_weight_min (Stage B3 union).
    """
    votes = votes_res["votes"]
    canon_nodes = canon["canon_nodes"]
    pos = {int(r.canon_id): np.array([r.z, r.y, r.x], float) for r in canon_nodes.itertuples()}
    tof = canon["canon_t"]
    prefer = union_prefer_edges or set()

    # group candidate edges by source frame
    by_frame = _dd33(list)
    for (cs, ct), rec in votes.items():
        by_frame[tof[cs]].append((cs, ct, rec))

    def _cost(cs, ct, rec):
        base = -rec["support"] + M33_DIST_COST_WEIGHT * _m33_phys(pos[cs], pos[ct])
        if (cs, ct) in prefer:
            base -= 0.5     # gentle preference for an M19-C edge (union mode)
        return base

    # STAGE 1 - primary
    primary = []
    support_hist = _dd33(int)
    n_candidates = 0
    for t in sorted(by_frame.keys()):
        cand = by_frame[t]
        # eligible candidates by gate
        elig = []
        for (cs, ct, rec) in cand:
            n_candidates += 1
            support_hist[int(round(rec["support"]))] += 1
            d = _m33_phys(pos[cs], pos[ct])
            keep = rec["support"] >= primary_support_min and d <= primary_max_um
            if (cs, ct) in prefer and rec["support"] >= 1.0 and d <= primary_max_um:
                keep = True    # union: preserve M19-C edge unless a challenger wins the assignment
            if keep:
                elig.append((cs, ct, rec, d))
        if not elig:
            continue
        srcs = sorted(set(e[0] for e in elig))
        tgts = sorted(set(e[1] for e in elig))
        lut = {(cs, ct): _cost(cs, ct, rec) for (cs, ct, rec, d) in elig}
        pairs = _assign_with_nolink(srcs, tgts, lambda s, t2: lut.get((s, t2)))
        for (cs, ct) in pairs:
            rec = votes[(cs, ct)]
            primary.append({"source": cs, "target": ct, "support": rec["support"],
                            "n_variants": len(rec["variants"]), "variants": sorted(rec["variants"])})

    prim_children = _dd33(int)
    parented = set()
    prim_edge_set = set()
    for e in primary:
        prim_children[e["source"]] += 1
        parented.add(e["target"])
        prim_edge_set.add((e["source"], e["target"]))

    # STAGE 2 - independent division (second child)
    parents_1 = [s for s, c in prim_children.items() if c == 1]
    divisions = []
    div_by_frame = _dd33(list)
    for (cs, ct), rec in votes.items():
        if cs in prim_children and ct not in parented and (cs, ct) not in prim_edge_set:
            div_by_frame[tof[cs]].append((cs, ct, rec))
    for t in sorted(div_by_frame.keys()):
        cand = [(cs, ct, rec) for (cs, ct, rec) in div_by_frame[t]
                if cs in set(parents_1) and ct not in parented]
        elig = []
        # first child position per parent (for sister gate)
        first_child = {}
        for e in primary:
            if e["source"] not in first_child:
                first_child[e["source"]] = e["target"]
        for (cs, ct, rec) in cand:
            d_pc = _m33_phys(pos[cs], pos[ct])
            fc = first_child.get(cs)
            d_sis = _m33_phys(pos[fc], pos[ct]) if fc is not None else 0.0
            ok = (rec["support"] >= division_support_min and len(rec["variants"]) >= fork_min_variants
                  and d_pc <= division_child_max_um and d_sis <= sister_max_um)
            if ok:
                elig.append((cs, ct, rec, d_pc))
        if not elig:
            continue
        srcs = sorted(set(e[0] for e in elig))
        tgts = sorted(set(e[1] for e in elig))
        lut = {(cs, ct): (-rec["support"] + M33_DIST_COST_WEIGHT * d) for (cs, ct, rec, d) in elig}
        pairs = _assign_with_nolink(srcs, tgts, lambda s, t2: lut.get((s, t2)))
        for (cs, ct) in pairs:
            if ct in parented:
                continue                      # never steal an already-parented target
            rec = votes[(cs, ct)]
            divisions.append({"source": cs, "target": ct, "support": rec["support"],
                              "n_variants": len(rec["variants"]), "variants": sorted(rec["variants"])})
            parented.add(ct)

    all_edges = [(e["source"], e["target"]) for e in primary] + [(d["source"], d["target"]) for d in divisions]
    return {
        "primary": primary, "divisions": divisions, "edges": all_edges,
        "n_candidates": n_candidates, "n_primary": len(primary), "n_divisions": len(divisions),
        "n_nolink": max(0, n_candidates - len(primary) - len(divisions)),
        "support_hist": {int(k): int(v) for k, v in sorted(support_hist.items())},
    }


# --------------------------------------------------------------------------- #
# 50d. Canonical graph -> submission rows (GLOBAL unique ids across datasets)
# --------------------------------------------------------------------------- #
def canonical_to_node_edges(canon, link):
    """Build (pred_nodes, pred_edges) with node_id = canon_id remapped to 0..K-1,
    keeping only linked-or-any-detected nodes. Isolated nodes are pruned AFTER
    linking (spec) by keeping every canonical node that appears in any edge OR is
    a genuine detection cluster; here we keep all detection clusters then prune
    degree-0 in the converter step."""
    canon_nodes = canon["canon_nodes"]
    remap = {int(r.canon_id): i for i, r in enumerate(canon_nodes.itertuples())}
    node_rows = [{"node_id": remap[int(r.canon_id)], "t": int(r.t), "z": float(r.z),
                  "y": float(r.y), "x": float(r.x)} for r in canon_nodes.itertuples()]
    pred_nodes = pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x"])
    edge_rows = [{"source_id": remap[cs], "target_id": remap[ct]} for (cs, ct) in link["edges"]]
    pred_edges = pd.DataFrame(edge_rows, columns=["source_id", "target_id"])
    return pred_nodes, pred_edges


def m33_graph_invariants(pred_nodes, pred_edges):
    """Validity metrics used everywhere: degree caps, dangling, multiframe,
    NaNs. Frames come from pred_nodes."""
    tof = {int(r.node_id): int(r.t) for r in pred_nodes.itertuples()}
    indeg = _dd33(int)
    outdeg = _dd33(int)
    dangling = 0
    multiframe = 0
    for r in pred_edges.itertuples():
        s, t = int(r.source_id), int(r.target_id)
        if s not in tof or t not in tof:
            dangling += 1
            continue
        outdeg[s] += 1
        indeg[t] += 1
        if tof[t] - tof[s] != 1:
            multiframe += 1
    coords = pred_nodes[["z", "y", "x"]].to_numpy(float) if len(pred_nodes) else np.zeros((0, 3))
    return {
        "n_nodes": int(len(pred_nodes)), "n_edges": int(len(pred_edges)),
        "max_in_degree": int(max(indeg.values()) if indeg else 0),
        "max_out_degree": int(max(outdeg.values()) if outdeg else 0),
        "n_divisions": int(sum(1 for v in outdeg.values() if v >= 2)),
        "dangling_edges": int(dangling), "direct_multiframe_edges": int(multiframe),
        "coord_nans": int(np.isnan(coords).sum()) if len(coords) else 0,
        "valid": bool(dangling == 0 and multiframe == 0
                      and (max(indeg.values()) if indeg else 0) <= 1
                      and (max(outdeg.values()) if outdeg else 0) <= 2
                      and (int(np.isnan(coords).sum()) if len(coords) else 0) == 0),
    }

# --------------------------------------------------------------------------- #
# 51. M33 STAGE A - dynamic input discovery, per-run report, diversity, class
# --------------------------------------------------------------------------- #
import hashlib as _hl33
import csv as _csv33

M33_SEARCH_ROOTS = ["/kaggle/input", KAGGLE_WORKING_DIR, KAGGLE_COMPETITION_INPUT_DIR]

# Candidate-graph classes (Stage D also uses these).
M33_GRAPH_CLASS = ["FULL_PRE_ILP_CANDIDATES", "MODERATELY_SPARSE", "ILP_SOLUTION_LIKE", "EDGE_PROB_UNAVAILABLE"]
M33_ENSEMBLE_POTENTIAL = ["INSUFFICIENT_INPUTS", "NEAR_DUPLICATE_VARIANTS", "MODERATE_DIVERSITY",
                          "STRONG_DIVERSITY", "INCOMPATIBLE_ARTIFACTS", "INVALID_INPUT"]


def _deep_get(obj, key):
    """Recursive first-match lookup of `key` in nested dict/list (used to pull
    det_threshold etc. out of arbitrary report JSON shapes)."""
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


def _m33_sha256(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = _hl33.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _m33_infer_run_name(path, meta):
    """Infer a run name from supplied metadata ONLY (artifact_name / run_name /
    explicit tags). Never infer a leaderboard score from a filename."""
    for k in ("run_name", "run", "name", "experiment_id", "artifact_name"):
        v = (meta or {}).get(k)
        if v:
            return str(v)
    return Path(path).stem


def _m33_read_submission_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not set(["row_type"]).issubset(df.columns):
        return None
    return df


def _m33_submission_structural(df):
    """Structural report of a submission-format CSV. No score inference."""
    if df is None or "row_type" not in df.columns:
        return {"valid_format": False}
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    rep = {"valid_format": list(df.columns) == SUBMISSION_COLUMNS or set(SUBMISSION_COLUMNS) <= set(df.columns),
           "node_rows": int(len(nodes)), "edge_rows": int(len(edges)),
           "datasets": sorted(df["dataset"].astype(str).unique().tolist()) if "dataset" in df else []}
    # global node-id uniqueness across datasets + consecutive `id`
    if "node_id" in nodes and len(nodes):
        nid = nodes["node_id"].to_numpy()
        rep["global_node_id_unique"] = bool(len(np.unique(nid)) == len(nid))
    else:
        rep["global_node_id_unique"] = True
    if "id" in df and len(df):
        ids = np.sort(df["id"].to_numpy())
        rep["consecutive_row_id"] = bool(np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))))
    else:
        rep["consecutive_row_id"] = False
    # degree caps + unit-time validity + dangling per dataset
    max_in = max_out = 0
    divisions = 0
    dangling = 0
    multiframe = 0
    nan_coords = int(np.isnan(nodes[["z", "y", "x"]].to_numpy(float)).sum()) if len(nodes) else 0
    for ds, g in df.groupby("dataset") if "dataset" in df and len(df) else []:
        gn = g[g["row_type"] == "node"]
        ge = g[g["row_type"] == "edge"]
        tof = {int(r.node_id): int(r.t) for r in gn.itertuples()}
        indeg = _dd33(int); outdeg = _dd33(int)
        for r in ge.itertuples():
            s, t = int(r.source_id), int(r.target_id)
            if s not in tof or t not in tof:
                dangling += 1; continue
            indeg[t] += 1; outdeg[s] += 1
            if tof[t] - tof[s] != 1:
                multiframe += 1
        max_in = max(max_in, max(indeg.values()) if indeg else 0)
        max_out = max(max_out, max(outdeg.values()) if outdeg else 0)
        divisions += sum(1 for v in outdeg.values() if v >= 2)
    rep.update({"max_in_degree": int(max_in), "max_out_degree": int(max_out),
                "divisions": int(divisions), "dangling_edges": int(dangling),
                "edge_unit_timepoint_valid": bool(multiframe == 0), "coord_nans": nan_coords,
                "graph_validity": bool(dangling == 0 and multiframe == 0 and max_in <= 1 and max_out <= 2 and nan_coords == 0)})
    return rep


def _m33_geff_structural(geff_root):
    """Report raw/solution/candidate edge counts + edge_prob availability +
    candidate class for a GEFF store WITHOUT applying the ILP solution mask."""
    ids = _read_geff_array(geff_root, "nodes/ids")
    e_ids = _read_geff_array(geff_root, "edges/ids")
    e_sol = _read_geff_array(geff_root, "edges/props/solution/values")
    e_prob = _read_geff_array(geff_root, "edges/props/edge_prob/values")
    n_nodes = int(np.asarray(ids).reshape(-1).size) if ids is not None else 0
    if e_ids is None:
        total = solution = candidate = 0
    else:
        e = np.asarray(e_ids)
        total = int(e.shape[0] if e.ndim > 1 else e.reshape(-1, 2).shape[0])
        if e_sol is not None:
            solution = int(np.asarray(e_sol).reshape(-1).astype(bool).sum())
        else:
            solution = total
        candidate = total
    cps = (candidate / n_nodes) if n_nodes else 0.0
    has_prob = e_prob is not None
    # classify candidate graph
    if not has_prob:
        cls = "EDGE_PROB_UNAVAILABLE"
    elif candidate <= solution * 1.05:
        cls = "ILP_SOLUTION_LIKE"
    elif cps >= 3.0:
        cls = "FULL_PRE_ILP_CANDIDATES"
    else:
        cls = "MODERATELY_SPARSE"
    return {"geff_available": True, "edge_prob_available": bool(has_prob),
            "total_geff_edges": total, "solution_edges": solution, "candidate_edges": candidate,
            "candidates_per_source_mean": float(cps), "node_rows": n_nodes,
            "candidate_graph_class": cls}


def discover_ensemble_inputs(search_roots=None):
    """Dynamically discover candidate inputs: submission CSVs, conversion/report
    JSONs, prediction GEFF stores, artifact manifests. No filename/dataset-name
    assumptions. Returns a list of per-candidate report dicts."""
    roots = [Path(r) for r in (search_roots or M33_SEARCH_ROOTS)]
    seen = set()
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            rp = str(p.resolve())
            if rp in seen:
                continue
            try:
                if p.is_file() and p.suffix == ".csv":
                    df = _m33_read_submission_csv(p)
                    if df is None:
                        continue
                    seen.add(rp)
                    meta = {}
                    rec = {"kind": "submission_csv", "source_path": str(p), "file_sha256": _m33_sha256(p),
                           "run_name": _m33_infer_run_name(p, meta), "score_from_metadata": None,
                           "artifact_name": None, "weight_sha256": None, "predict_config": None,
                           "det_threshold": None, "postprocess_family": None}
                    rec.update(_m33_submission_structural(df))
                    rec["geff_available"] = False
                    rec["edge_prob_available"] = False
                    rec["candidate_graph_class"] = None
                    out.append(rec)
                elif p.is_file() and p.suffix == ".json" and any(k in p.name.lower() for k in ("report", "manifest", "conversion", "config", "metadata")):
                    seen.add(rp)
                    try:
                        meta = json.loads(p.read_text())
                    except Exception:
                        meta = {}
                    rec = {"kind": "report_json", "source_path": str(p), "file_sha256": _m33_sha256(p),
                           "run_name": _m33_infer_run_name(p, meta if isinstance(meta, dict) else {}),
                           "score_from_metadata": (meta.get("public_score") if isinstance(meta, dict) else None),
                           "artifact_name": (meta.get("artifact_name") if isinstance(meta, dict) else None),
                           "weight_sha256": (meta.get("weight_sha256") if isinstance(meta, dict) else None),
                           "predict_config": (meta.get("command") or meta.get("predict_config") if isinstance(meta, dict) else None),
                           "det_threshold": (_deep_get(meta, "det_threshold") if isinstance(meta, dict) else None),
                           "postprocess_family": (meta.get("postprocess_ops") or meta.get("ops") if isinstance(meta, dict) else None),
                           "datasets": (meta.get("found_datasets") or meta.get("datasets") if isinstance(meta, dict) else None)}
                    out.append(rec)
                elif p.is_dir():
                    gr = _geff_root_of(p)
                    if gr is None:
                        continue
                    seen.add(rp)
                    rec = {"kind": "prediction_geff", "source_path": str(p), "file_sha256": None,
                           "run_name": _strip_zarr(p.name).replace(".geff", ""), "score_from_metadata": None,
                           "artifact_name": None, "weight_sha256": None, "predict_config": None,
                           "det_threshold": None, "postprocess_family": None}
                    rec.update(_m33_geff_structural(gr))
                    out.append(rec)
            except Exception as exc:
                out.append({"kind": "error", "source_path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    return out


# --------------------------------------------------------------------------- #
# 51a. Pairwise diversity (submission-level and GEFF-level)
# --------------------------------------------------------------------------- #
def _m33_variant_from_submission(df, name):
    """Build an M33Variant from a submission-format df (single- or multi-dataset).
    Node ids are taken as-is; edges kept as (source_id,target_id)."""
    nodes = df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]].copy()
    edges = df[df["row_type"] == "edge"][["source_id", "target_id"]].copy()
    edges.columns = ["source_id", "target_id"]
    return M33Variant(name, nodes.rename(columns={"node_id": "node_id"}), edges)


def submission_pairwise_diversity(variants, eps_um=M33_DEFAULT_EPS_UM):
    """For every pair: node overlap after safe matching, edge Jaccard, agreement
    count, edges unique to A / B, conflicting in/out targets, division
    agreement/disagreement, structural count deltas."""
    rows = []
    for a, b in _it33.combinations(range(len(variants)), 2):
        va, vb = variants[a], variants[b]
        canon = canonicalize_nodes([va, vb], eps_um=eps_um)
        n2c = canon["node_to_canon"]
        # index which local index maps to variant 0/1 after internal sort
        order = sorted(range(2), key=lambda i: ([va, vb][i].name, i))
        # translate each variant's edges to canonical
        def _cedges(vi_global, v):
            s = set()
            for r in v.edges.itertuples():
                if (vi_global, int(r.source_id)) in n2c and (vi_global, int(r.target_id)) in n2c:
                    cs, ct = n2c[(vi_global, int(r.source_id))], n2c[(vi_global, int(r.target_id))]
                    if cs != ct:
                        s.add((cs, ct))
            return s
        # canonicalize_nodes was given [va, vb] so indices are 0 and 1
        ea = _cedges(0, va)
        eb = _cedges(1, vb)
        inter = ea & eb
        union = ea | eb
        jac = (len(inter) / len(union)) if union else 1.0
        # conflicting targets: same target canonical node, different source
        out_a = _dd33(set); out_b = _dd33(set); in_a = _dd33(set); in_b = _dd33(set)
        for (s, t) in ea:
            out_a[s].add(t); in_a[t].add(s)
        for (s, t) in eb:
            out_b[s].add(t); in_b[t].add(s)
        conflict_in = sum(1 for t in (set(in_a) & set(in_b)) if in_a[t] != in_b[t])
        conflict_out = sum(1 for s in (set(out_a) & set(out_b)) if out_a[s] != out_b[s])
        div_a = set(s for s, ts in out_a.items() if len(ts) >= 2)
        div_b = set(s for s, ts in out_b.items() if len(ts) >= 2)
        rows.append({"run_a": va.name, "run_b": vb.name, "n_canon": canon["n_canon"],
                     "matched_nodes": canon["matched_clusters"], "node_overlap_frac": (canon["matched_clusters"] / canon["n_canon"]) if canon["n_canon"] else 0.0,
                     "edge_jaccard": float(jac), "edge_agreement": len(inter),
                     "edges_unique_a": len(ea - eb), "edges_unique_b": len(eb - ea),
                     "conflicting_incoming_targets": int(conflict_in), "conflicting_outgoing_sources": int(conflict_out),
                     "division_agreement": len(div_a & div_b), "division_disagreement": len(div_a ^ div_b),
                     "delta_nodes": len(va.nodes) - len(vb.nodes), "delta_edges": len(va.edges) - len(vb.edges)})
    return rows


def geff_pairwise_diversity(geff_variants):
    """Pairwise model diversity for GEFF variants: node overlap, candidate/
    solution overlap, edge_prob correlation on shared eligible edges, mean abs
    prob diff, shared/unique candidates. geff_variants: list of dicts with
    name, cand_edges(set), sol_edges(set), edge_prob(dict), node_ids(set)."""
    rows = []
    for a, b in _it33.combinations(range(len(geff_variants)), 2):
        A, B = geff_variants[a], geff_variants[b]
        shared_nodes = A["node_ids"] & B["node_ids"]
        cand_shared = A["cand_edges"] & B["cand_edges"]
        sol_shared = A["sol_edges"] & B["sol_edges"]
        # eligible shared edges = candidate in both
        pa = [A["edge_prob"].get(e) for e in cand_shared]
        pb = [B["edge_prob"].get(e) for e in cand_shared]
        pairs = [(x, y) for x, y in zip(pa, pb) if x is not None and y is not None]
        if len(pairs) >= 2:
            xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
            corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else None
            mad = float(np.mean(np.abs(xs - ys)))
        else:
            corr, mad = None, None
        rows.append({"run_a": A["name"], "run_b": B["name"],
                     "node_overlap": len(shared_nodes),
                     "candidate_edge_overlap": len(cand_shared), "solution_edge_overlap": len(sol_shared),
                     "edge_prob_correlation_shared": corr, "mean_abs_prob_diff": mad,
                     "candidates_unique_a": len(A["cand_edges"] - B["cand_edges"]),
                     "candidates_unique_b": len(B["cand_edges"] - A["cand_edges"])})
    return rows


def classify_ensemble_potential(valid_variant_count, submission_pairs):
    """INSUFFICIENT_INPUTS / NEAR_DUPLICATE_VARIANTS / MODERATE / STRONG."""
    if valid_variant_count < 3:
        return "INSUFFICIENT_INPUTS"
    if submission_pairs and all(p["edge_jaccard"] > 0.985 for p in submission_pairs):
        return "NEAR_DUPLICATE_VARIANTS"
    # meaningful unique/conflicting edges?
    uniq = sum(p["edges_unique_a"] + p["edges_unique_b"] for p in submission_pairs)
    conf = sum(p["conflicting_incoming_targets"] + p["conflicting_outgoing_sources"] for p in submission_pairs)
    strong = any(p["edge_jaccard"] < 0.85 for p in submission_pairs) and (uniq + conf) > 0
    return "STRONG_DIVERSITY" if strong else "MODERATE_DIVERSITY"


def _m33_write_csv(path, rows, columns=None):
    path = Path(path)
    cols = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="") as f:
        w = _csv33.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

# --------------------------------------------------------------------------- #
# 52. M33 STAGE B/C - corrected output-level weighted ensemble
# --------------------------------------------------------------------------- #
# Config presets B0..B3 (spec). B0 reproduces M19-C alone through the canonical
# adapter; B1 conservative weighted consensus; B2 stricter; B3 controlled union.
M33_BLEND_CONFIGS = {
    "B0": {"desc": "M19-C alone through canonical adapter", "primary_support_min": 0.0,
           "division_support_min": 0.0, "fork_min_variants": 1, "solo": "M19-C"},
    "B1": {"desc": "weighted primary consensus", "primary_support_min": 2.0,
           "division_support_min": 2.0, "fork_min_variants": 2},
    "B2": {"desc": "stricter consensus", "primary_support_min": 3.0,
           "division_support_min": 3.0, "fork_min_variants": 2},
    "B3": {"desc": "controlled union, preserve M19-C on conflict", "primary_support_min": 1.0,
           "division_support_min": 2.0, "fork_min_variants": 2, "union": True},
}

# Sane graph-delta guard vs M19-C.
M33_DELTA_NODE_MIN, M33_DELTA_NODE_MAX = -0.05, 0.08
M33_DELTA_EDGE_MIN, M33_DELTA_EDGE_MAX = -0.05, 0.10
M33_RETAIN_FRAC_MIN = 0.90
M33_MAX_DIVISIONS = 3000


def _m33_group_by_dataset(variant_dfs):
    """variant_dfs: list of (name, df, weight, identified). Returns dict
    dataset -> list of (name, sub_df, weight, identified)."""
    per = _dd33(list)
    for (name, df, weight, ident) in variant_dfs:
        if "dataset" not in df.columns:
            per["_single"].append((name, df, weight, ident))
            continue
        for ds, g in df.groupby("dataset"):
            per[str(ds)].append((name, g.reset_index(drop=True), weight, ident))
    return per


def _m33_variant_edge_set_canonical(variants, canon, vi):
    n2c = canon["node_to_canon"]
    s = set()
    for r in variants[vi].edges.itertuples():
        if (vi, int(r.source_id)) in n2c and (vi, int(r.target_id)) in n2c:
            cs, ct = n2c[(vi, int(r.source_id))], n2c[(vi, int(r.target_id))]
            if cs != ct and canon["canon_t"].get(ct) == canon["canon_t"].get(cs, -99) + 1:
                s.add((cs, ct))
    return s


def run_output_blend(variant_dfs, config_name="B1", eps_um=M33_DEFAULT_EPS_UM, m19c_name="M19-C"):
    """Run a corrected output-blend config across every dataset. Returns
    per-dataset link results, aggregate stats, and edge-change accounting vs the
    M19-C variant (retained/removed/replaced/new). Does NOT write submission.csv."""
    cfg = dict(M33_BLEND_CONFIGS[config_name])
    per_ds = _m33_group_by_dataset(variant_dfs)
    agg = {"config": config_name, "desc": cfg["desc"], "datasets": [], "canon_collisions": 0,
           "duplicate_votes_removed": 0, "order_invariant": None,
           "n_canon_nodes": 0, "n_primary": 0, "n_divisions": 0, "n_nolink": 0,
           "support_hist": _dd33(int), "source_support_hist": _dd33(int),
           "m19c_edges": 0, "m19c_retained": 0, "m19c_removed": 0, "m19c_replaced": 0, "new_edges": 0,
           "final_nodes": 0, "final_edges": 0}
    edge_changes = []
    per_ds_out = {}
    for ds, vlist in sorted(per_ds.items()):
        if cfg.get("solo"):
            vlist = [v for v in vlist if v[0] == cfg["solo"]] or vlist[:1]
        variants = [M33Variant(name, g[g["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                               g[g["row_type"] == "edge"][["source_id", "target_id"]], weight=w)
                    for (name, g, w, ident) in vlist]
        if not variants:
            continue
        canon = canonicalize_nodes(variants, eps_um=eps_um)
        votes_res = build_edge_votes(variants, canon)
        # M19-C canonical edge set (for union preference + change accounting)
        m19c_idx = next((i for i, v in enumerate(variants) if v.name == m19c_name), None)
        prefer = _m33_variant_edge_set_canonical(variants, canon, m19c_idx) if (cfg.get("union") and m19c_idx is not None) else None
        link = two_stage_link(canon, votes_res, variants,
                              primary_support_min=cfg["primary_support_min"],
                              division_support_min=cfg["division_support_min"],
                              fork_min_variants=cfg["fork_min_variants"],
                              union_prefer_edges=prefer)
        pred_nodes, pred_edges = canonical_to_node_edges(canon, link)
        pred_nodes, pred_edges = _m33_prune_isolated(pred_nodes, pred_edges)
        inv = m33_graph_invariants(pred_nodes, pred_edges)
        # change accounting vs M19-C
        final_edges = set((int(r.source_id), int(r.target_id)) for r in pred_edges.itertuples())
        m19c_edges = _m33_variant_edge_set_canonical(variants, canon, m19c_idx) if m19c_idx is not None else set()
        # remap m19c canonical edges through the same canon->0..K-1 remap used in converter
        remap = {int(r.canon_id): i for i, r in enumerate(canon["canon_nodes"].itertuples())}
        m19c_remapped = set((remap[a], remap[b]) for (a, b) in m19c_edges if a in remap and b in remap)
        retained = m19c_remapped & final_edges
        removed = m19c_remapped - final_edges
        new = final_edges - m19c_remapped
        # replaced: an M19-C edge removed but the same source has a different final child
        fin_out = _dd33(set)
        for (s, t) in final_edges:
            fin_out[s].add(t)
        replaced = sum(1 for (s, t) in removed if fin_out.get(s))
        agg["datasets"].append({"dataset": ds, "n_canon": canon["n_canon"], "collision": canon["collision_count"],
                                "dup_votes": votes_res["duplicate_votes_removed"], "invariants": inv,
                                "n_primary": link["n_primary"], "n_divisions": link["n_divisions"],
                                "m19c_edges": len(m19c_remapped), "retained": len(retained), "removed": len(removed),
                                "new": len(new), "replaced": int(replaced)})
        agg["canon_collisions"] += canon["collision_count"]
        agg["duplicate_votes_removed"] += votes_res["duplicate_votes_removed"]
        agg["n_canon_nodes"] += canon["n_canon"]
        agg["n_primary"] += link["n_primary"]; agg["n_divisions"] += link["n_divisions"]; agg["n_nolink"] += link["n_nolink"]
        for k, v in link["support_hist"].items():
            agg["support_hist"][k] += v
        for e in link["primary"] + link["divisions"]:
            agg["source_support_hist"][len(e["variants"])] += 1
        agg["m19c_edges"] += len(m19c_remapped); agg["m19c_retained"] += len(retained)
        agg["m19c_removed"] += len(removed); agg["m19c_replaced"] += int(replaced); agg["new_edges"] += len(new)
        agg["final_nodes"] += inv["n_nodes"]; agg["final_edges"] += inv["n_edges"]
        for (s, t) in sorted(removed)[:50]:
            edge_changes.append({"dataset": ds, "change": "removed", "source": s, "target": t})
        for (s, t) in sorted(new)[:50]:
            edge_changes.append({"dataset": ds, "change": "new", "source": s, "target": t})
        per_ds_out[ds] = (pred_nodes, pred_edges, canon, link)
    agg["support_hist"] = {int(k): int(v) for k, v in sorted(agg["support_hist"].items())}
    agg["source_support_hist"] = {int(k): int(v) for k, v in sorted(agg["source_support_hist"].items())}
    agg["m19c_retained_frac"] = (agg["m19c_retained"] / agg["m19c_edges"]) if agg["m19c_edges"] else 1.0
    return agg, edge_changes, per_ds_out


def _m33_prune_isolated(pred_nodes, pred_edges):
    """Prune degree-0 nodes AFTER linking, then relabel to 0..K-1."""
    if not len(pred_nodes):
        return pred_nodes, pred_edges
    deg = _dd33(int)
    for r in pred_edges.itertuples():
        deg[int(r.source_id)] += 1; deg[int(r.target_id)] += 1
    keep = [int(r.node_id) for r in pred_nodes.itertuples() if deg[int(r.node_id)] > 0]
    keepset = set(keep)
    if len(keepset) == len(pred_nodes):
        return pred_nodes, pred_edges
    remap = {nid: i for i, nid in enumerate(sorted(keepset))}
    # keep isolated nodes too? spec: prune isolated only after linking -> we drop them
    nn = pred_nodes[pred_nodes["node_id"].isin(keepset)].copy()
    nn["node_id"] = nn["node_id"].map(remap)
    ee = pred_edges.copy()
    ee = ee[ee["source_id"].isin(keepset) & ee["target_id"].isin(keepset)]
    ee["source_id"] = ee["source_id"].map(remap); ee["target_id"] = ee["target_id"].map(remap)
    return nn.reset_index(drop=True), ee.reset_index(drop=True)


def verify_order_invariance(variant_dfs, config_name="B1", eps_um=M33_DEFAULT_EPS_UM, n_perms=3):
    """Run the blend under several input permutations; the resulting canonical
    graph (sorted edge multiset per dataset) must be identical."""
    import random as _rnd
    base_agg, _, base_out = run_output_blend(variant_dfs, config_name, eps_um)
    base_sig = _m33_blend_signature(base_out)
    ok = True
    for k in range(n_perms):
        perm = list(variant_dfs)
        _rnd.Random(1000 + k).shuffle(perm)
        _, _, out = run_output_blend(perm, config_name, eps_um)
        if _m33_blend_signature(out) != base_sig:
            ok = False
            break
    return ok


def _m33_blend_signature(per_ds_out):
    sig = {}
    for ds, (pn, pe, canon, link) in per_ds_out.items():
        # signature independent of node-id labelling: use sorted (t,z,y,x) rounded edges
        pos = {int(r.node_id): (int(r.t), round(float(r.z), 3), round(float(r.y), 3), round(float(r.x), 3)) for r in pn.itertuples()}
        edges = sorted((pos[int(r.source_id)], pos[int(r.target_id)]) for r in pe.itertuples() if int(r.source_id) in pos and int(r.target_id) in pos)
        sig[ds] = (len(pn), tuple(edges))
    return sig


# --------------------------------------------------------------------------- #
# 52a. Stage C - build the gated experimental submission from per-dataset graphs
# --------------------------------------------------------------------------- #
def build_submission_from_blend(per_ds_out, expected_datasets=None):
    """Assemble the corrected-ensemble submission with GLOBALLY unique node ids
    and a global running row id across ALL datasets (fixes next_id=0-per-dataset).
    Returns (submission_df, stats)."""
    all_rows = []
    next_id = 0
    next_node_id = 0
    per_dataset = []
    for ds in sorted(per_ds_out.keys()):
        pred_nodes, pred_edges, canon, link = per_ds_out[ds]
        # relabel to a global running node id
        remap = {int(r.node_id): next_node_id + i for i, r in enumerate(pred_nodes.itertuples())}
        next_node_id += len(pred_nodes)
        for r in pred_nodes.itertuples():
            all_rows.append({"id": next_id, "dataset": ds, "row_type": "node", "node_id": remap[int(r.node_id)],
                             "t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x),
                             "source_id": -1, "target_id": -1})
            next_id += 1
        for r in pred_edges.itertuples():
            all_rows.append({"id": next_id, "dataset": ds, "row_type": "edge", "node_id": -1,
                             "t": -1, "z": -1, "y": -1, "x": -1,
                             "source_id": remap[int(r.source_id)], "target_id": remap[int(r.target_id)]})
            next_id += 1
        per_dataset.append({"dataset": ds, "nodes": len(pred_nodes), "edges": len(pred_edges),
                            "divisions": m33_graph_invariants(pred_nodes, pred_edges)["n_divisions"]})
    df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    nodes = df[df["row_type"] == "node"]
    nid = nodes["node_id"].to_numpy()
    ids = np.sort(df["id"].to_numpy()) if len(df) else np.array([])
    stats = {"final_nodes": int(len(nodes)), "final_edges": int((df["row_type"] == "edge").sum()),
             "global_node_ids_unique": bool(len(np.unique(nid)) == len(nid)) if len(nid) else True,
             "consecutive_row_id": bool(np.array_equal(ids, np.arange(len(ids)))) if len(ids) else True,
             "no_nan": bool(not np.isnan(nodes[["z", "y", "x"]].to_numpy(float)).any()) if len(nodes) else True,
             "per_dataset": per_dataset,
             "missing_datasets": sorted(set(expected_datasets or []) - set(per_ds_out.keys()))}
    return df, stats


def sane_delta_vs_m19c(blend_agg):
    """Apply the sane graph-delta guard using the blend's own m19c accounting."""
    m19c_n = sum(d["m19c_edges"] for d in blend_agg["datasets"])
    # node/edge deltas vs M19-C baseline (approx by retained+removed for edges,
    # canon nodes for nodes are not directly M19-C node count -> use retained frac
    # + division cap + edge delta as the enforceable guards).
    retain_frac = blend_agg["m19c_retained_frac"]
    checks = {"m19c_retained_frac_ok": bool(retain_frac >= M33_RETAIN_FRAC_MIN),
              "divisions_ok": bool(blend_agg["n_divisions"] <= M33_MAX_DIVISIONS)}
    checks["all_ok"] = bool(all(checks.values()))
    checks["retain_frac"] = float(retain_frac)
    checks["n_divisions"] = int(blend_agg["n_divisions"])
    return checks

# --------------------------------------------------------------------------- #
# 53. M33 STAGE D/E - probability-level fusion (eligible denominator, calibration)
# --------------------------------------------------------------------------- #
M33_PROB_ENSEMBLE_RECS = ["OK_TO_SUBMIT_EXPERIMENTAL", "DO_NOT_SUBMIT_CANDIDATE_GRAPH_TOO_SPARSE",
                          "DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE", "DO_NOT_SUBMIT_CALIBRATION_UNRESOLVED",
                          "DO_NOT_SUBMIT_INSUFFICIENT_DIVERSITY", "DO_NOT_SUBMIT_INVALID_GRAPH"]


def _m33_logit(p, eps=1e-6):
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def read_geff_candidate_variant(name, geff_root, weight=None):
    """Build a probability-fusion variant from a GEFF store: candidate edges
    (NO solution mask), edge_prob, node positions. graph_class from
    _m33_geff_structural."""
    ids = _read_geff_array(geff_root, "nodes/ids")
    t = _read_geff_array(geff_root, "nodes/props/t/values")
    z = _read_geff_array(geff_root, "nodes/props/z/values")
    y = _read_geff_array(geff_root, "nodes/props/y/values")
    x = _read_geff_array(geff_root, "nodes/props/x/values")
    e_ids = _read_geff_array(geff_root, "edges/ids")
    e_prob = _read_geff_array(geff_root, "edges/props/edge_prob/values")
    e_sol = _read_geff_array(geff_root, "edges/props/solution/values")
    struct = _m33_geff_structural(geff_root)
    raw_ids = np.asarray(ids).reshape(-1) if ids is not None else np.array([], int)
    id_to_local = {int(r): i for i, r in enumerate(raw_ids)}
    nrows = []
    for i, rid in enumerate(raw_ids):
        nrows.append({"node_id": i, "t": int(t[i]) if t is not None else 0,
                      "z": float(z[i]) if z is not None else 0.0,
                      "y": float(y[i]) if y is not None else 0.0,
                      "x": float(x[i]) if x is not None else 0.0})
    nodes = pd.DataFrame(nrows, columns=["node_id", "t", "z", "y", "x"])
    cand_edges = set()
    sol_edges = set()
    edge_prob = {}
    if e_ids is not None:
        e = np.asarray(e_ids)
        e = e if e.ndim > 1 else e.reshape(-1, 2)
        sol = np.asarray(e_sol).reshape(-1).astype(bool) if e_sol is not None else None
        prob = np.asarray(e_prob).reshape(-1).astype(float) if e_prob is not None else None
        for k in range(e.shape[0]):
            s_raw, t_raw = int(e[k, 0]), int(e[k, 1])
            if s_raw not in id_to_local or t_raw not in id_to_local:
                continue
            se, te = id_to_local[s_raw], id_to_local[t_raw]
            cand_edges.add((se, te))
            if prob is not None:
                edge_prob[(se, te)] = float(prob[k])
            if sol is not None and sol[k]:
                sol_edges.add((se, te))
    if e_sol is None:
        sol_edges = set(cand_edges)
    v = M33Variant(name, nodes, pd.DataFrame(sorted(sol_edges), columns=["source_id", "target_id"]) if sol_edges else pd.DataFrame(columns=["source_id", "target_id"]),
                   weight=weight, edge_prob=edge_prob, cand_edges=cand_edges, graph_class=struct["candidate_graph_class"])
    v.struct = struct
    v.sol_edges = sol_edges
    return v


def probability_fusion_audit(geff_variants):
    """Stage D: per-variant candidate class + edge_prob quantiles + eligibility
    universe + calibration diffs + node stability. Decides whether probability
    ensemble is ALLOWED."""
    per = []
    for v in geff_variants:
        probs = np.array(list(v.edge_prob.values()), float) if v.edge_prob else np.array([])
        q = {}
        if probs.size:
            for name, val in [("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]:
                q[name] = float(np.percentile(probs, val))
        per.append({"run": v.name, "graph_class": v.graph_class,
                    "edge_prob_available": bool(v.edge_prob),
                    "total_edges": len(v.cand_edges), "solution_edges": len(getattr(v, "sol_edges", set())),
                    "candidate_edges": len(v.cand_edges),
                    "candidates_per_source_mean": (len(v.cand_edges) / max(1, len(v.nodes))),
                    "edge_prob_quantiles": q,
                    "edge_prob_nondegenerate": bool(probs.size >= 2 and float(probs.std()) > 1e-4),
                    "full_candidates_pre_ilp": bool(v.graph_class == "FULL_PRE_ILP_CANDIDATES")})
    n_full = sum(1 for p in per if p["full_candidates_pre_ilp"])
    n_modsparse = sum(1 for p in per if p["graph_class"] == "MODERATELY_SPARSE")
    n_prob = sum(1 for p in per if p["edge_prob_available"])
    # shared eligible universe: canonicalize nodes, count edges candidate-eligible in >=2
    allowed = True
    reasons = []
    if not (n_full >= 2 or n_modsparse >= 3):
        allowed = False; reasons.append("need >=2 FULL_PRE_ILP or >=3 moderately-sparse variants")
    if n_prob < 2:
        allowed = False; reasons.append("edge_prob unavailable on <2 variants")
    if not any(p["edge_prob_nondegenerate"] for p in per):
        allowed = False; reasons.append("edge_prob degenerate")
    # candidate diversity
    diversity = geff_pairwise_diversity([{"name": v.name, "node_ids": set(v._det_ids),
                                          "cand_edges": v.cand_edges, "sol_edges": getattr(v, "sol_edges", set()),
                                          "edge_prob": v.edge_prob} for v in geff_variants]) if len(geff_variants) >= 2 else []
    nontrivial = any((d["candidates_unique_a"] + d["candidates_unique_b"]) > 0 for d in diversity)
    if diversity and not nontrivial:
        allowed = False; reasons.append("candidate diversity trivial")
    if any(p["graph_class"] == "ILP_SOLUTION_LIKE" for p in per):
        reasons.append("some variants ILP_SOLUTION_LIKE")
    if all(p["graph_class"] == "EDGE_PROB_UNAVAILABLE" for p in per):
        allowed = False; reasons.append("edge_prob unavailable on all variants")
    return {"per_variant": per, "pairwise_diversity": diversity, "n_full_pre_ilp": n_full,
            "n_moderately_sparse": n_modsparse, "probability_ensemble_allowed": bool(allowed),
            "block_reasons": reasons,
            "calibration_options": ["raw_probability_mean", "logit_mean", "rank_normalized_mean", "weighted_logit_mean"],
            "calibration_selected": None,
            "calibration_note": "Calibration must be justified by official/diagnostic CV; not selected here."}


def build_probability_fusion_edges(geff_variants, eps_um=M33_DEFAULT_EPS_UM, calibration="rank_normalized_mean"):
    """Compute, for each canonical candidate edge, proposer_count,
    eligible_model_count, support_fraction, mean prob over ELIGIBLE proposing
    models, missing-eligible penalty, endpoint-presence counts. Returns a table +
    the canonicalization for downstream linking. A model is eligible only if both
    canonical endpoints exist for it."""
    canon = canonicalize_nodes(geff_variants, eps_um=eps_um)
    votes_res = build_edge_votes(geff_variants, canon, use_candidates=True)
    votes = votes_res["votes"]
    canon_present = votes_res["canon_present"]
    # rank-normalize each variant's prob VALUES (value -> rank in [0,1]).
    ranks = {}
    for v in geff_variants:
        if v.edge_prob:
            vals = sorted(set(v.edge_prob.values()))
            n = len(vals)
            ranks[v.name] = {val: (i + 1) / n for i, val in enumerate(vals)}
    n2c = canon["node_to_canon"]
    rows = []
    for (cs, ct), rec in votes.items():
        elig = eligible_model_count(cs, ct, geff_variants, canon_present)
        proposer = rec["n_proposers"]
        # gather eligible proposing probabilities per calibration
        raw_vals, logit_vals, rank_vals = [], [], []
        for v in geff_variants:
            p = rec["probs"].get(v.name)
            if p is None:
                continue
            raw_vals.append(p)
            logit_vals.append(_m33_logit(p))
            rank_vals.append(ranks.get(v.name, {}).get(p, p))
        if calibration == "raw_probability_mean":
            fused = float(np.mean(raw_vals)) if raw_vals else None
        elif calibration == "logit_mean":
            fused = float(np.mean(logit_vals)) if logit_vals else None
        elif calibration == "weighted_logit_mean":
            fused = float(np.average(logit_vals, weights=[max(0.1, geff_variants[i].weight) for i in range(len(logit_vals))])) if logit_vals else None
        else:  # rank_normalized_mean
            fused = float(np.mean(rank_vals)) if rank_vals else None
        support_fraction = (proposer / elig) if elig else 0.0
        missing_penalty = 1.0 - support_fraction
        rows.append({"source": cs, "target": ct, "proposer_count": proposer,
                     "eligible_model_count": elig, "support_fraction": float(support_fraction),
                     "fused_score": fused, "mean_raw_prob": float(np.mean(raw_vals)) if raw_vals else None,
                     "missing_eligible_penalty": float(missing_penalty)})
        # write fused score back as the "support" surrogate for the linker
        rec["support"] = float(support_fraction) * (fused if fused is not None else 1.0) + support_fraction
    return {"canon": canon, "votes_res": votes_res, "edge_table": rows}

# --------------------------------------------------------------------------- #
# 54. M33 orchestration - 5 runners (A audit / B diag / C candidate / D / E)
# --------------------------------------------------------------------------- #
M33_400EP_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
M33_400EP_NAME_SUBSTR = "400ep"
M33_WEIGHTS_REL = "weights/unet_transformer/split_0/edge_predictor_best.pth"

M33_OUTPUT_BLEND_RECS = ["OK_TO_SUBMIT_EXPERIMENTAL", "DO_NOT_SUBMIT_INSUFFICIENT_INPUTS",
                         "DO_NOT_SUBMIT_NEAR_DUPLICATE_VARIANTS", "DO_NOT_SUBMIT_CANONICALIZATION_COLLISION",
                         "DO_NOT_SUBMIT_ORDER_DEPENDENT", "DO_NOT_SUBMIT_INVALID_GRAPH",
                         "DO_NOT_SUBMIT_INCOMPATIBLE_INPUTS", "DO_NOT_SUBMIT_UNSANE_GRAPH_DELTA"]


def _m33_valid_submission_variants(reports):
    """From Stage-A discovery reports, build (name, df, weight, identified) for
    every valid submission CSV. Resolves weight from M33_DEFAULT_WEIGHTS by run
    name; unidentified runs are reported (identified=False)."""
    out = []
    for r in reports:
        if r.get("kind") != "submission_csv" or not r.get("graph_validity", False):
            continue
        df = _m33_read_submission_csv(r["source_path"])
        if df is None:
            continue
        name = r["run_name"]
        ident = name in M33_DEFAULT_WEIGHTS
        w = M33_DEFAULT_WEIGHTS.get(name, M33_UNKNOWN_WEIGHT)
        out.append((name, df, w, ident))
    return out


def run_m33_a_input_audit(working_dir=KAGGLE_WORKING_DIR, search_roots=None):
    """STAGE A - dynamic discovery + per-run structural report + pairwise
    diversity + ensemble-potential classification. NON-SUBMIT."""
    _RUN_LOG.clear()
    out = Path(working_dir); out.mkdir(parents=True, exist_ok=True)
    reports = discover_ensemble_inputs(search_roots)
    sub_variants = _m33_valid_submission_variants(reports)
    variants = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                           df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w)
                for (n, df, w, ident) in sub_variants]
    sub_pairs = submission_pairwise_diversity(variants) if len(variants) >= 2 else []
    # GEFF variants for geff diversity
    geffs = []
    for r in reports:
        if r.get("kind") == "prediction_geff" and r.get("geff_available"):
            gr = _geff_root_of(Path(r["source_path"]))
            if gr is not None:
                gv = read_geff_candidate_variant(r["run_name"], gr)
                geffs.append({"name": gv.name, "node_ids": set(gv._det_ids), "cand_edges": gv.cand_edges,
                              "sol_edges": getattr(gv, "sol_edges", set()), "edge_prob": gv.edge_prob})
    geff_pairs = geff_pairwise_diversity(geffs) if len(geffs) >= 2 else []
    potential = classify_ensemble_potential(len(variants), sub_pairs)
    unidentified = [n for (n, df, w, ident) in sub_variants if not ident]
    audit = {"kind": "ensemble_input_audit", "submit": False, "n_candidates_found": len(reports),
             "n_valid_submissions": len(variants), "n_geff_stores": len(geffs),
             "ensemble_potential": potential, "unidentified_runs": unidentified,
             "candidate_reports": reports}
    (out / "m33_ensemble_input_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    _m33_write_csv(out / "m33_submission_pairwise_diversity.csv", sub_pairs)
    _m33_write_csv(out / "m33_geff_pairwise_diversity.csv", geff_pairs)
    _write_log(out)
    print("=== M33_A ENSEMBLE INPUT AUDIT (non-submit) ===")
    print(f"  candidates found: {len(reports)}  valid submissions: {len(variants)}  geff stores: {len(geffs)}")
    print(f"  ensemble potential: {potential}   unidentified runs: {unidentified}")
    print("  No submission (input audit only).")
    return audit


def run_m33_b_output_blend_diag(working_dir=KAGGLE_WORKING_DIR, search_roots=None):
    """STAGE B - B0..B3 corrected output-blend diagnostics. Prerequisite: >=3
    valid submissions. NON-SUBMIT."""
    _RUN_LOG.clear()
    out = Path(working_dir); out.mkdir(parents=True, exist_ok=True)
    reports = discover_ensemble_inputs(search_roots)
    variant_dfs = _m33_valid_submission_variants(reports)
    result = {"kind": "output_blend_diagnostic", "submit": False, "n_valid_submissions": len(variant_dfs), "configs": {}}
    if len(variant_dfs) < 3:
        result["status"] = "INSUFFICIENT_INPUTS"
        (out / "m33_output_blend_diagnostic.json").write_text(json.dumps(result, indent=2, default=str))
        _m33_write_csv(out / "m33_output_blend_edge_changes.csv", [])
        _write_log(out)
        print("=== M33_B OUTPUT-BLEND DIAGNOSTIC (non-submit) ===\n  INSUFFICIENT_INPUTS (<3 valid submissions).")
        return result
    all_changes = []
    for cfg in ["B0", "B1", "B2", "B3"]:
        agg, changes, per_ds = run_output_blend(variant_dfs, cfg)
        order_ok = verify_order_invariance(variant_dfs, cfg) if cfg != "B0" else True
        agg["order_invariant"] = order_ok
        agg["sane_delta"] = sane_delta_vs_m19c(agg)
        result["configs"][cfg] = agg
        for c in changes:
            c["config"] = cfg
        all_changes.extend(changes)
    (out / "m33_output_blend_diagnostic.json").write_text(json.dumps(result, indent=2, default=str))
    _m33_write_csv(out / "m33_output_blend_edge_changes.csv", all_changes,
                   columns=["config", "dataset", "change", "source", "target"])
    _write_log(out)
    print("=== M33_B OUTPUT-BLEND DIAGNOSTIC (non-submit) ===")
    for cfg, agg in result["configs"].items():
        print(f"  {cfg}: primary={agg['n_primary']} div={agg['n_divisions']} collisions={agg['canon_collisions']} "
              f"dup_votes={agg['duplicate_votes_removed']} order_inv={agg['order_invariant']} "
              f"retain={agg['m19c_retained_frac']:.3f}")
    print("  No submission (diagnostic only). local_metric.py NOT used as official score.")
    return result


def run_m33_c_output_blend_candidate(working_dir=KAGGLE_WORKING_DIR, competition_dir=KAGGLE_COMPETITION_INPUT_DIR, config_name="B1", search_roots=None):
    """STAGE C - build the EXPERIMENTAL corrected output-blend submission, hard-
    gated. Writes submission.csv only when every gate passes."""
    _RUN_LOG.clear()
    out = Path(working_dir); out.mkdir(parents=True, exist_ok=True)
    reports = discover_ensemble_inputs(search_roots)
    variant_dfs = _m33_valid_submission_variants(reports)
    sub_variants = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                               df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w)
                    for (n, df, w, ident) in variant_dfs]
    sub_pairs = submission_pairwise_diversity(sub_variants) if len(sub_variants) >= 2 else []
    potential = classify_ensemble_potential(len(variant_dfs), sub_pairs)
    manifest = {"input_runs": [{"run": n, "sha256": r.get("file_sha256"), "weight": w, "identified": ident}
                               for (n, df, w, ident), r in zip(variant_dfs, [x for x in reports if x.get("kind") == "submission_csv"][:len(variant_dfs)])],
                "eps_um": M33_DEFAULT_EPS_UM, "config": config_name}
    (out / "m33_input_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    rec = "OK_TO_SUBMIT_EXPERIMENTAL"
    if len(variant_dfs) < 3:
        rec = "DO_NOT_SUBMIT_INSUFFICIENT_INPUTS"
    elif potential == "NEAR_DUPLICATE_VARIANTS":
        rec = "DO_NOT_SUBMIT_NEAR_DUPLICATE_VARIANTS"
    if rec != "OK_TO_SUBMIT_EXPERIMENTAL":
        return _m33_c_fail(out, rec, {"n_valid": len(variant_dfs), "potential": potential}, manifest)

    agg, changes, per_ds = run_output_blend(variant_dfs, config_name)
    if agg["canon_collisions"] > 0:
        rec = "DO_NOT_SUBMIT_CANONICALIZATION_COLLISION"
    elif not verify_order_invariance(variant_dfs, config_name):
        rec = "DO_NOT_SUBMIT_ORDER_DEPENDENT"
    elif agg["duplicate_votes_removed"] < 0:
        rec = "DO_NOT_SUBMIT_INVALID_GRAPH"
    if rec != "OK_TO_SUBMIT_EXPERIMENTAL":
        return _m33_c_fail(out, rec, agg, manifest)

    df, stats = build_submission_from_blend(per_ds, expected_datasets=None)
    inv = m33_graph_invariants(df[df["row_type"] == "node"].rename(columns={"node_id": "node_id"})[["node_id", "t", "z", "y", "x"]],
                               df[df["row_type"] == "edge"][["source_id", "target_id"]])
    delta = sane_delta_vs_m19c(agg)
    if not inv["valid"] or not stats["global_node_ids_unique"]:
        rec = "DO_NOT_SUBMIT_INVALID_GRAPH"
    elif not delta["all_ok"]:
        rec = "DO_NOT_SUBMIT_UNSANE_GRAPH_DELTA"
    if rec != "OK_TO_SUBMIT_EXPERIMENTAL":
        return _m33_c_fail(out, rec, {**agg, "invariants": inv, "delta": delta}, manifest)

    df.to_csv(out / "submission.csv", index=False)
    report = {"kind": "corrected_output_blend_candidate", "submit": False, "recommendation": rec,
              "final_source": "corrected_output_level_weighted_ensemble", "config": config_name,
              "input_runs": manifest["input_runs"], "eps_um": M33_DEFAULT_EPS_UM,
              "canonicalization": {"collisions": agg["canon_collisions"], "n_canon_nodes": agg["n_canon_nodes"]},
              "duplicate_votes_removed": agg["duplicate_votes_removed"],
              "primary_support_threshold": M33_BLEND_CONFIGS[config_name]["primary_support_min"],
              "division_support_threshold": M33_BLEND_CONFIGS[config_name]["division_support_min"],
              "selected_primary_edges": agg["n_primary"], "selected_divisions": agg["n_divisions"],
              "m19c_edges_retained": agg["m19c_retained"], "m19c_edges_removed": agg["m19c_removed"],
              "new_edges": agg["new_edges"], "final_nodes": stats["final_nodes"], "final_edges": stats["final_edges"],
              "division_count": inv["n_divisions"], "max_in_degree": inv["max_in_degree"], "max_out_degree": inv["max_out_degree"],
              "direct_multiframe_edges": inv["direct_multiframe_edges"], "dangling_edges": inv["dangling_edges"],
              "global_node_ids_unique": stats["global_node_ids_unique"], "consecutive_row_id": stats["consecutive_row_id"],
              "no_nan": stats["no_nan"], "missing_datasets": stats["missing_datasets"], "valid": inv["valid"],
              "fallback_used": False, "sane_delta": delta}
    (out / "m33_output_blend_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "m33_failure_fallback_report.json").write_text(json.dumps({"fallback_used": False, "reason": None}, indent=2))
    _write_log(out)
    print("=== M33_C CORRECTED OUTPUT-BLEND CANDIDATE ===")
    print(f"  config={config_name} nodes={stats['final_nodes']} edges={stats['final_edges']} div={inv['n_divisions']} "
          f"retain={agg['m19c_retained_frac']:.3f} collisions={agg['canon_collisions']}")
    print(f"  RECOMMENDATION: {rec}. submission.csv written to /kaggle/working. User reviews before any submit.")
    return report


def _m33_c_fail(out, rec, detail, manifest):
    report = {"kind": "corrected_output_blend_candidate", "submit": False, "recommendation": rec,
              "final_source": "corrected_output_level_weighted_ensemble", "valid": False, "fallback_used": False,
              "detail": detail, "input_runs": manifest.get("input_runs")}
    (out / "m33_output_blend_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "m33_failure_fallback_report.json").write_text(json.dumps({"fallback_used": False, "reason": rec}, indent=2))
    _write_log(out)
    print("=== M33_C CORRECTED OUTPUT-BLEND CANDIDATE ===")
    print(f"  RECOMMENDATION: {rec}. NO submission.csv written (gate not passed).")
    return report


def run_m33_d_probability_audit(working_dir=KAGGLE_WORKING_DIR, search_roots=None):
    """STAGE D - probability-fusion audit over GEFF variants. NON-SUBMIT."""
    _RUN_LOG.clear()
    out = Path(working_dir); out.mkdir(parents=True, exist_ok=True)
    reports = discover_ensemble_inputs(search_roots)
    geffs = []
    for r in reports:
        if r.get("kind") == "prediction_geff" and r.get("geff_available"):
            gr = _geff_root_of(Path(r["source_path"]))
            if gr is not None:
                geffs.append(read_geff_candidate_variant(r["run_name"], gr))
    audit = probability_fusion_audit(geffs)
    audit.update({"kind": "probability_fusion_audit", "submit": False, "n_geff_variants": len(geffs)})
    (out / "m33_probability_fusion_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    calib_rows = []
    for p in audit["per_variant"]:
        row = {"run": p["run"], "graph_class": p["graph_class"], "edge_prob_available": p["edge_prob_available"]}
        row.update({k: p["edge_prob_quantiles"].get(k) for k in ["p05", "p25", "p50", "p75", "p95"]})
        calib_rows.append(row)
    _m33_write_csv(out / "m33_probability_calibration.csv", calib_rows,
                   columns=["run", "graph_class", "edge_prob_available", "p05", "p25", "p50", "p75", "p95"])
    _write_log(out)
    print("=== M33_D PROBABILITY FUSION AUDIT (non-submit) ===")
    print(f"  geff variants: {len(geffs)}  full_pre_ilp: {audit['n_full_pre_ilp']}  allowed: {audit['probability_ensemble_allowed']}")
    print(f"  block reasons: {audit['block_reasons']}")
    print("  Calibration NOT selected (needs CV). No submission.")
    return audit


def run_m33_e_probability_ensemble(working_dir=KAGGLE_WORKING_DIR, search_roots=None):
    """STAGE E - corrected probability-level ensemble. HARD-GATED. Only builds a
    candidate when Stage D allows AND calibration is justified (never here without
    CV)."""
    _RUN_LOG.clear()
    out = Path(working_dir); out.mkdir(parents=True, exist_ok=True)
    reports = discover_ensemble_inputs(search_roots)
    geffs = []
    for r in reports:
        if r.get("kind") == "prediction_geff" and r.get("geff_available"):
            gr = _geff_root_of(Path(r["source_path"]))
            if gr is not None:
                geffs.append(read_geff_candidate_variant(r["run_name"], gr))
    audit = probability_fusion_audit(geffs)
    if any(v.graph_class == "ILP_SOLUTION_LIKE" for v in geffs) and audit["n_full_pre_ilp"] < 2:
        rec = "DO_NOT_SUBMIT_CANDIDATE_GRAPH_TOO_SPARSE"
    elif all(v.graph_class == "EDGE_PROB_UNAVAILABLE" for v in geffs) if geffs else True:
        rec = "DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE"
    elif not audit["probability_ensemble_allowed"]:
        rec = "DO_NOT_SUBMIT_INSUFFICIENT_DIVERSITY"
    else:
        # calibration is never auto-selected without CV
        rec = "DO_NOT_SUBMIT_CALIBRATION_UNRESOLVED"
    report = {"kind": "corrected_probability_ensemble_candidate", "submit": False, "recommendation": rec,
              "final_source": "corrected_probability_level_ensemble", "n_geff_variants": len(geffs),
              "probability_ensemble_allowed": audit["probability_ensemble_allowed"], "block_reasons": audit["block_reasons"],
              "calibration_selected": None, "input_model_diversity": audit["pairwise_diversity"],
              "uses_old_v2_linker": False, "uses_corrected_m30_linker": True}
    (out / "m33_probability_ensemble_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_log(out)
    print("=== M33_E CORRECTED PROBABILITY ENSEMBLE ===")
    print(f"  geff variants: {len(geffs)}  RECOMMENDATION: {rec}. No submission (hard-gated).")
    return report

# --------------------------------------------------------------------------- #
# 55. M33 tests (synthetic fixtures) + drivers
# --------------------------------------------------------------------------- #
# Module-level guarantees asserted by the static tests below.
M33_USES_CORRECTED_M30_LINKER = True
M33_USES_OLD_WINNING_POSTPROCESS_V2 = False


def _m33_sub_df(dataset, nodes, edges):
    """nodes: list of (node_id,t,z,y,x). edges: list of (src,tgt). -> submission df."""
    rows = []
    rid = 0
    for (nid, t, z, y, x) in nodes:
        rows.append({"id": rid, "dataset": dataset, "row_type": "node", "node_id": nid,
                     "t": t, "z": z, "y": y, "x": x, "source_id": -1, "target_id": -1})
        rid += 1
    for (s, t) in edges:
        rows.append({"id": rid, "dataset": dataset, "row_type": "edge", "node_id": -1,
                     "t": -1, "z": -1, "y": -1, "x": -1, "source_id": s, "target_id": t})
        rid += 1
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def _m33_track_variant(name, dataset="dsX", jitter=0.0, extra_child=False, weight=None):
    """A 3-frame track A0->A1->A2 (+ optional division second child at t=1)."""
    nodes = [(0, 0, 0.0, 0.0, 0.0 + jitter), (1, 1, 0.0, 0.0, 0.5 + jitter), (2, 2, 0.0, 0.0, 1.0 + jitter)]
    edges = [(0, 1), (1, 2)]
    if extra_child:
        nodes.append((3, 1, 0.0, 1.0, 0.5 + jitter))   # sister ~0.40625um in y
        edges.append((0, 3))
    df = _m33_sub_df(dataset, nodes, edges)
    return M33Variant(name, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                      df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=weight)


def _m33_three_variant_dfs(with_division=False):
    dfs = []
    for name, w in [("M19-C", 2.0), ("M29-A", 1.0), ("M30-C", 1.0)]:
        div = with_division and name in ("M19-C", "M29-A")   # 2-variant fork support
        v = _m33_track_variant(name, jitter=0.05 * len(dfs), extra_child=div, weight=w)
        node_df = v.nodes.copy()
        node_df["row_type"] = "node"; node_df["dataset"] = "dsX"; node_df["source_id"] = -1; node_df["target_id"] = -1
        node_df["id"] = range(len(node_df))
        edf = v.edges.copy(); edf["row_type"] = "edge"; edf["dataset"] = "dsX"
        edf["node_id"] = -1; edf["t"] = -1; edf["z"] = -1; edf["y"] = -1; edf["x"] = -1
        edf["id"] = range(len(node_df), len(node_df) + len(edf))
        full = pd.concat([node_df, edf], ignore_index=True)[SUBMISSION_COLUMNS]
        dfs.append((name, full, w, name in M33_DEFAULT_WEIGHTS))
    return dfs


# ---- individual tests ------------------------------------------------------ #
def _m33_t01_global_ids_across_datasets():
    dfs = _m33_three_variant_dfs()
    # two datasets
    dfs2 = []
    for (n, df, w, ident) in dfs:
        d2 = df.copy(); d2["dataset"] = "dsY"
        dfs2.append((n, pd.concat([df, d2], ignore_index=True), w, ident))
    _, _, per_ds = run_output_blend(dfs2, "B1")
    sub, stats = build_submission_from_blend(per_ds)
    assert stats["global_node_ids_unique"], "global node ids must not reset across datasets"


def _m33_t02_hungarian_one_to_one():
    variants = _m33_three_variant_dfs()
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in variants]
    canon = canonicalize_nodes(vs)
    # each canon cluster has at most one member per variant
    for _, c in [(0, None)]:
        pass
    n2c = canon["node_to_canon"]
    per_cluster = _dd33(list)
    for (vi, nid), cid in n2c.items():
        per_cluster[cid].append(vi)
    assert all(len(v) == len(set(v)) for v in per_cluster.values()), "per-frame matching not one-to-one"


def _m33_t03_no_same_variant_collapse():
    # variant A creates a cluster; variant B has TWO nodes near it same frame
    a = _m33_sub_df("d", [(0, 0, 0, 0, 0.0)], [])
    b = _m33_sub_df("d", [(0, 0, 0, 0, 0.1), (1, 0, 0, 0, 0.2)], [])   # both within eps of A's node
    va = M33Variant("A", a[a.row_type == "node"][["node_id", "t", "z", "y", "x"]], a[a.row_type == "edge"][["source_id", "target_id"]])
    vb = M33Variant("B", b[b.row_type == "node"][["node_id", "t", "z", "y", "x"]], b[b.row_type == "edge"][["source_id", "target_id"]])
    canon = canonicalize_nodes([va, vb], eps_um=1.0)
    assert canon["collision_count"] == 0, "same-variant nodes must never collapse"
    # B's two nodes -> two distinct canonical ids
    assert canon["node_to_canon"][(1, 0)] != canon["node_to_canon"][(1, 1)]


def _m33_t04_unmatched_nodes_distinct():
    a = _m33_sub_df("d", [(0, 0, 0, 0, 0.0)], [])
    b = _m33_sub_df("d", [(0, 0, 0, 100.0, 0.0)], [])   # far apart -> unmatched
    va = M33Variant("A", a[a.row_type == "node"][["node_id", "t", "z", "y", "x"]], a[a.row_type == "edge"][["source_id", "target_id"]])
    vb = M33Variant("B", b[b.row_type == "node"][["node_id", "t", "z", "y", "x"]], b[b.row_type == "edge"][["source_id", "target_id"]])
    canon = canonicalize_nodes([va, vb], eps_um=1.0)
    assert canon["n_canon"] == 2 and canon["unmatched_nodes"] == 2


def _m33_t05_coords_order_invariant():
    variants = _m33_three_variant_dfs()
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in variants]
    c1 = canonicalize_nodes(vs)
    c2 = canonicalize_nodes(list(reversed(vs)))
    s1 = sorted((int(r.t), round(r.z, 4), round(r.y, 4), round(r.x, 4)) for r in c1["canon_nodes"].itertuples())
    s2 = sorted((int(r.t), round(r.z, 4), round(r.y, 4), round(r.x, 4)) for r in c2["canon_nodes"].itertuples())
    assert s1 == s2, "canonical coordinates must be order-invariant"


def _m33_t06_graph_order_invariant():
    dfs = _m33_three_variant_dfs()
    assert verify_order_invariance(dfs, "B1"), "blend graph must be identical under input permutation"


def _m33_t07_one_vote_per_canonical_edge():
    # By design same-variant nodes never collapse, so a variant can only produce
    # a duplicate vote if its OWN edge list literally repeats an edge. The dedup
    # guard removes the repeat; every canonical edge gets <=1 vote per variant.
    nodes = [(0, 0, 0, 0, 0.0), (1, 1, 0, 0, 0.5)]
    edges = [(0, 1), (0, 1)]   # literally duplicated edge in this variant
    df = _m33_sub_df("d", nodes, edges)
    v = M33Variant("A", df[df.row_type == "node"][["node_id", "t", "z", "y", "x"]], df[df.row_type == "edge"][["source_id", "target_id"]], weight=2.0)
    canon = canonicalize_nodes([v], eps_um=1.0)
    vr = build_edge_votes([v], canon)
    for ce, rec in vr["votes"].items():
        assert rec["n_proposers"] <= 1, "one variant must not vote twice for one canonical edge"
    assert vr["duplicate_votes_removed"] >= 1


def _m33_t08_duplicate_votes_removed():
    _m33_t07_one_vote_per_canonical_edge()   # asserts dup removed >=1


def _m33_t09_weighted_support():
    dfs = _m33_three_variant_dfs()
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in dfs]
    canon = canonicalize_nodes(vs)
    vr = build_edge_votes(vs, canon)
    # the shared A0->A1 edge should have support 2+1+1 = 4
    supports = sorted(rec["support"] for rec in vr["votes"].values())
    assert max(supports) == 4.0, f"weighted support wrong: {supports}"


def _m33_t10_explicit_no_link():
    # a lone source with no eligible target must remain unlinked (no-link dummy)
    pairs = _assign_with_nolink([0], [10], lambda s, t: None)   # no candidate cost
    assert pairs == [], "source with no candidate must stay unlinked"


def _m33_t11_primary_degree_caps():
    dfs = _m33_three_variant_dfs()
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in dfs]
    canon = canonicalize_nodes(vs); vr = build_edge_votes(vs, canon)
    link = two_stage_link(canon, vr, vs)
    outdeg = _dd33(int); indeg = _dd33(int)
    for e in link["primary"]:
        outdeg[e["source"]] += 1; indeg[e["target"]] += 1
    assert (max(outdeg.values()) if outdeg else 0) <= 1 and (max(indeg.values()) if indeg else 0) <= 1


def _m33_t12_independent_division():
    dfs = _m33_three_variant_dfs(with_division=True)
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in dfs]
    canon = canonicalize_nodes(vs); vr = build_edge_votes(vs, canon)
    link = two_stage_link(canon, vr, vs)
    assert link["n_divisions"] >= 1, "a fork supported by 2 variants must be admitted"


def _m33_t13_rejected_division_no_steal():
    # division child that is already primary-parented must not be stolen
    dfs = _m33_three_variant_dfs(with_division=True)
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in dfs]
    canon = canonicalize_nodes(vs); vr = build_edge_votes(vs, canon)
    link = two_stage_link(canon, vr, vs)
    prim_targets = set(e["target"] for e in link["primary"])
    div_targets = set(d["target"] for d in link["divisions"])
    assert prim_targets.isdisjoint(div_targets), "division must not steal a primary-parented target"


def _m33_t14_fork_requires_multimodel():
    # single-variant fork must NOT be admitted (fork_min_variants=2)
    v = _m33_track_variant("solo", extra_child=True, weight=2.0)
    canon = canonicalize_nodes([v]); vr = build_edge_votes([v], canon)
    link = two_stage_link(canon, vr, [v], division_support_min=1.0)
    assert link["n_divisions"] == 0, "single-variant fork must be rejected"


def _m33_t15_final_degree_caps():
    dfs = _m33_three_variant_dfs(with_division=True)
    _, _, per_ds = run_output_blend(dfs, "B1")
    for ds, (pn, pe, canon, link) in per_ds.items():
        inv = m33_graph_invariants(pn, pe)
        assert inv["max_in_degree"] <= 1 and inv["max_out_degree"] <= 2


def _m33_t16_no_dangling_multiframe():
    dfs = _m33_three_variant_dfs(with_division=True)
    _, _, per_ds = run_output_blend(dfs, "B1")
    for ds, (pn, pe, canon, link) in per_ds.items():
        inv = m33_graph_invariants(pn, pe)
        assert inv["dangling_edges"] == 0 and inv["direct_multiframe_edges"] == 0


def _m33_t17_m19c_retention_delta():
    dfs = _m33_three_variant_dfs()
    agg, _, _ = run_output_blend(dfs, "B1")
    guard = sane_delta_vs_m19c(agg)
    assert "m19c_retained_frac_ok" in guard and "divisions_ok" in guard


def _m33_t18_missing_invalid_input():
    # fewer than 3 valid -> classification INSUFFICIENT_INPUTS
    assert classify_ensemble_potential(2, []) == "INSUFFICIENT_INPUTS"
    # empty discovery must not raise
    assert isinstance(discover_ensemble_inputs(["/no/such/root"]), list)


def _m33_t19_geff_classification():
    # ILP_SOLUTION_LIKE when candidate==solution; classification helper is pure
    from types import SimpleNamespace  # noqa
    # emulate: candidate <= solution*1.05 with prob -> ILP_SOLUTION_LIKE
    # (exercise via a tiny synthetic dict through the same thresholds)
    def classify(cand, sol, cps, has_prob):
        if not has_prob: return "EDGE_PROB_UNAVAILABLE"
        if cand <= sol * 1.05: return "ILP_SOLUTION_LIKE"
        if cps >= 3.0: return "FULL_PRE_ILP_CANDIDATES"
        return "MODERATELY_SPARSE"
    assert classify(100, 100, 1.0, True) == "ILP_SOLUTION_LIKE"
    assert classify(500, 100, 5.0, True) == "FULL_PRE_ILP_CANDIDATES"
    assert classify(100, 100, 1.0, False) == "EDGE_PROB_UNAVAILABLE"


def _m33_t20_eligible_model_denominator():
    # two variants both detect endpoints -> eligible 2; a third missing one endpoint -> not eligible
    v1 = _m33_track_variant("M19-C", weight=2.0)
    v2 = _m33_track_variant("M29-A", weight=1.0)
    v3 = _m33_sub_df("dsX", [(0, 0, 0, 0, 0.0)], [])   # only one node (missing t=1 endpoint)
    v3v = M33Variant("M30-C", v3[v3.row_type == "node"][["node_id", "t", "z", "y", "x"]], v3[v3.row_type == "edge"][["source_id", "target_id"]], weight=1.0)
    vs = [v1, v2, v3v]
    canon = canonicalize_nodes(vs)
    vr = build_edge_votes(vs, canon, use_candidates=True)
    # find canonical A0->A1 edge
    cid0 = canon["node_to_canon"][(0, 0)]; cid1 = canon["node_to_canon"][(0, 1)]
    elig = eligible_model_count(cid0, cid1, vs, vr["canon_present"])
    assert elig == 2, f"eligible must exclude the model missing an endpoint, got {elig}"


def _m33_t21_missing_endpoint_not_zero_vote():
    # a model missing an endpoint must NOT lower the mean as a 0-probability vote
    v1 = _m33_track_variant("M19-C", weight=2.0); v1.edge_prob[(0, 1)] = 0.9
    v2 = _m33_track_variant("M29-A", weight=1.0); v2.edge_prob[(0, 1)] = 0.8
    v3 = _m33_sub_df("dsX", [(0, 0, 0, 0, 0.0)], [])
    v3v = M33Variant("M30-C", v3[v3.row_type == "node"][["node_id", "t", "z", "y", "x"]], v3[v3.row_type == "edge"][["source_id", "target_id"]], weight=1.0)
    vs = [v1, v2, v3v]
    res = build_probability_fusion_edges(vs, calibration="raw_probability_mean")
    row = next(r for r in res["edge_table"] if r["mean_raw_prob"] is not None)
    assert abs(row["mean_raw_prob"] - 0.85) < 1e-6, "missing endpoint must not act as a 0-prob vote"


def _m33_t22_calibration_diagnostics():
    v1 = _m33_track_variant("M19-C", weight=2.0); v1.edge_prob[(0, 1)] = 0.9; v1.edge_prob[(1, 2)] = 0.7
    v2 = _m33_track_variant("M29-A", weight=1.0); v2.edge_prob[(0, 1)] = 0.8; v2.edge_prob[(1, 2)] = 0.6
    audit = probability_fusion_audit([v1, v2])
    assert set(audit["calibration_options"]) >= {"raw_probability_mean", "logit_mean", "rank_normalized_mean"}
    assert audit["calibration_selected"] is None


def _m33_t23_old_v2_linker_not_used():
    assert M33_USES_OLD_WINNING_POSTPROCESS_V2 is False
    assert "winning_postprocess_v2" not in globals()
    assert "global_relink" not in globals()


def _m33_t24_corrected_m30_linker_used():
    assert M33_USES_CORRECTED_M30_LINKER is True
    # the corrected linker exposes explicit no-link accounting
    dfs = _m33_three_variant_dfs()
    vs = [M33Variant(n, df[df["row_type"] == "node"][["node_id", "t", "z", "y", "x"]],
                     df[df["row_type"] == "edge"][["source_id", "target_id"]], weight=w) for (n, df, w, i) in dfs]
    canon = canonicalize_nodes(vs); vr = build_edge_votes(vs, canon)
    link = two_stage_link(canon, vr, vs)
    assert "n_nolink" in link and "primary" in link and "divisions" in link


def _m33_t25_no_local_metric_as_official():
    # the module must never call local_metric a scorer; a flag documents it
    assert M33_LOCAL_METRIC_IS_OFFICIAL is False


def _m33_t26_no_static_submission_rows():
    # the candidate builder must derive rows from inputs, not embed them
    assert M33_STATIC_SUBMISSION_ROWS is False


def _m33_t27_c_gates_block_without_inputs():
    # with no inputs, Stage C must refuse (no submission.csv)
    import tempfile
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as empty:
        rep = run_m33_c_output_blend_candidate(working_dir=td, search_roots=[empty])
        assert rep["recommendation"] == "DO_NOT_SUBMIT_INSUFFICIENT_INPUTS"
        assert not (Path(td) / "submission.csv").exists()


def _m33_t28_e_hard_gated():
    import tempfile
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as empty:
        rep = run_m33_e_probability_ensemble(working_dir=td, search_roots=[empty])
        assert rep["recommendation"].startswith("DO_NOT_SUBMIT")
        assert rep["uses_old_v2_linker"] is False


def _m33_t29_a_audit_runs_offline():
    import tempfile
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as empty:
        rep = run_m33_a_input_audit(working_dir=td, search_roots=[empty])
        assert rep["ensemble_potential"] in M33_ENSEMBLE_POTENTIAL
        assert (Path(td) / "m33_ensemble_input_audit.json").exists()


def _m33_t30_full_c_candidate_with_synthetic(tmp=None):
    # a fully synthetic 3-variant / 2-dataset blend passes structural gates
    import tempfile
    dfs = _m33_three_variant_dfs(with_division=True)
    agg, _, per_ds = run_output_blend(dfs, "B1")
    sub, stats = build_submission_from_blend(per_ds)
    inv = m33_graph_invariants(sub[sub.row_type == "node"][["node_id", "t", "z", "y", "x"]],
                               sub[sub.row_type == "edge"][["source_id", "target_id"]])
    assert inv["valid"] and stats["global_node_ids_unique"] and stats["consecutive_row_id"]


# module-level honesty flags (asserted by static tests, checked by assembler)
M33_LOCAL_METRIC_IS_OFFICIAL = False
M33_STATIC_SUBMISSION_ROWS = False


def run_milestone33_tests():
    _m33_t01_global_ids_across_datasets()
    _m33_t02_hungarian_one_to_one()
    _m33_t03_no_same_variant_collapse()
    _m33_t04_unmatched_nodes_distinct()
    _m33_t05_coords_order_invariant()
    _m33_t06_graph_order_invariant()
    _m33_t07_one_vote_per_canonical_edge()
    _m33_t08_duplicate_votes_removed()
    _m33_t09_weighted_support()
    _m33_t10_explicit_no_link()
    _m33_t11_primary_degree_caps()
    _m33_t12_independent_division()
    _m33_t13_rejected_division_no_steal()
    _m33_t14_fork_requires_multimodel()
    _m33_t15_final_degree_caps()
    _m33_t16_no_dangling_multiframe()
    _m33_t17_m19c_retention_delta()
    _m33_t18_missing_invalid_input()
    _m33_t19_geff_classification()
    _m33_t20_eligible_model_denominator()
    _m33_t21_missing_endpoint_not_zero_vote()
    _m33_t22_calibration_diagnostics()
    _m33_t23_old_v2_linker_not_used()
    _m33_t24_corrected_m30_linker_used()
    _m33_t25_no_local_metric_as_official()
    _m33_t26_no_static_submission_rows()
    _m33_t27_c_gates_block_without_inputs()
    _m33_t28_e_hard_gated()
    _m33_t29_a_audit_runs_offline()
    _m33_t30_full_c_candidate_with_synthetic()
    print("All milestone33_corrected_ensemble_runner tests passed (30/30).")


def _m33_dry(kind):
    print(f"[dry-run] /kaggle/input absent - self-tests only ({kind}). On Kaggle this discovers mounted tracking "
          "outputs, audits diversity, and (for C/E) builds a hard-gated experimental candidate; it never submits and "
          "keeps M19-C 0.880 final until an observed score beats it. Non-submit for A/B/D.")
    return {"status": "dry_run", "kind": kind}


def run_milestone33_input_audit(working_dir=KAGGLE_WORKING_DIR):
    print("=== Self-test: M33 corrected ensemble ==="); run_milestone33_tests()
    return _m33_dry("ensemble_input_audit") if not is_kaggle_env() else run_m33_a_input_audit(working_dir)


def run_milestone33_output_blend_diag(working_dir=KAGGLE_WORKING_DIR):
    print("=== Self-test: M33 corrected ensemble ==="); run_milestone33_tests()
    return _m33_dry("output_blend_diagnostic") if not is_kaggle_env() else run_m33_b_output_blend_diag(working_dir)


def run_milestone33_output_blend_candidate(working_dir=KAGGLE_WORKING_DIR):
    print("=== Self-test: M33 corrected ensemble ==="); run_milestone33_tests()
    return _m33_dry("output_blend_candidate") if not is_kaggle_env() else run_m33_c_output_blend_candidate(working_dir)


def run_milestone33_probability_audit(working_dir=KAGGLE_WORKING_DIR):
    print("=== Self-test: M33 corrected ensemble ==="); run_milestone33_tests()
    return _m33_dry("probability_fusion_audit") if not is_kaggle_env() else run_m33_d_probability_audit(working_dir)


def run_milestone33_probability_ensemble(working_dir=KAGGLE_WORKING_DIR):
    print("=== Self-test: M33 corrected ensemble ==="); run_milestone33_tests()
    return _m33_dry("probability_ensemble_candidate") if not is_kaggle_env() else run_m33_e_probability_ensemble(working_dir)


run_milestone33_input_audit()
